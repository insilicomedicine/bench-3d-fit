"""
Built-in reward functions.

Built-in scoring criteria are defined as :class:`RewardFunction` subclasses,
each overriding ``compute(mols, conditions)`` with its scoring logic.
Reward functions are **stateless** with respect to per-row context:
:class:`~bench_3d_fit.generation_conditions.GenerationConditions` carries
the seed molecule, receptor PDB, anchor points, etc. at call time.  Only
truly static state (loaded ML models, resolved executable paths, docking
parameters) lives on ``self``.

Subclasses are registered by name (see the ``register_reward`` calls at the
bottom of this module) for use with :class:`~bench_3d_fit.reward.scorer.RewardScorer`.
"""

import logging
import os
import subprocess
import tempfile
import traceback
from typing import List, Optional, Tuple
from pathlib import Path
from yaml import safe_load

_SHM_DIR = "/dev/shm" if os.path.isdir("/dev/shm") else None

import numpy as np
import prolif as plf
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, QED
from rdkit.Chem.rdMolDescriptors import (
    CalcExactMolWt,
    CalcNumLipinskiHBD,
    CalcNumLipinskiHBA,
    CalcCrippenDescriptors,
    CalcTPSA,
    DoubleCubicLatticeVolume
)
from espsim import GetEspSim, GetShapeSim
from posebusters import PoseBusters
from unidock_tools.application.ligandprep import ligprep

from bench_3d_fit.reward.base import RewardFunction, register_reward
from bench_3d_fit.reward.utils import (
    pdb_to_pdbqt_cached,
    resolve_unidock_executable,
    build_anchor_points_smarts,
)
from bench_3d_fit.reward.utils import (
    get_ligand_pli_points,
    score_pharmacophore_points,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core RewardFunction subclasses
# ---------------------------------------------------------------------------
class DescriptorsReward(RewardFunction):
    """Descriptors Reward.
    """
    name = "descriptors"
    SCORE_DESCRIPTIONS = {
        "nheavy": "Number of Heavy Atoms",
        "mw": "Molecular Weight",
        "hbd": "Hydrogen Bond Donor",
        "hba": "Hydrogen Bond Acceptor",
        "logp": "LogP",
        "tpsa": "Topological Polar Surface Area",
        "qed": "Quantitative Estimate of Druglikeness",
        "ro5_mw_passed": "Molecular Weight Passed",
        "ro5_hbd_passed": "Hydrogen Bond Donor Passed",
        "ro5_hba_passed": "Hydrogen Bond Acceptor Passed",
        "ro5_logp_passed": "LogP Passed",
        "ro5_passed": "All Rules of Five Passed",
    }
    def compute(self, mols, configuration):
        output = {
            "nheavy": [],
            "vdw_volume": [],
            "mw": [],
            "hbd": [],
            "hba": [],
            "logp": [],
            "tpsa": [],
            "qed": [],
            "ro5_mw_passed": [],
            "ro5_hbd_passed": [],
            "ro5_hba_passed": [],
            "ro5_logp_passed": [],
            "ro5_passed": [],
        }
        for mol in mols:
            if mol is None:
                for key in output.keys():
                    output[key].append(None)
                continue
            nheavy = mol.GetNumHeavyAtoms()
            volume_obj = DoubleCubicLatticeVolume(mol)
            volume = volume_obj.GetVDWVolume()
            mw = CalcExactMolWt(mol)
            hbd = CalcNumLipinskiHBD(mol)
            hba = CalcNumLipinskiHBA(mol)
            logp = CalcCrippenDescriptors(mol)[0]
            tpsa = CalcTPSA(mol)
            output["nheavy"].append(nheavy)
            output["vdw_volume"].append(volume)
            output["mw"].append(mw)
            output["hbd"].append(hbd)
            output["hba"].append(hba)
            output["logp"].append(logp)
            output["tpsa"].append(tpsa)
            output["qed"].append(QED.qed(mol))
            output["ro5_mw_passed"].append(mw < 500)
            output["ro5_hbd_passed"].append(hbd <= 5)
            output["ro5_hba_passed"].append(hba <= 10)
            output["ro5_logp_passed"].append(logp < 5)
            output["ro5_passed"].append(
                output["ro5_mw_passed"][-1] and output["ro5_hbd_passed"][-1] and \
                    output["ro5_hba_passed"][-1] and output["ro5_logp_passed"][-1]
            )
        return output


class PoseBustersReward(RewardFunction):
    """PoseBusters Reward.

    Requires a PoseBusters object (``pb_obj``) passed at construction
    time.  If ``pb_obj`` is ``None``, all scores are returned as
    ``None``.
    """

    name = "posebusters"

    SCORE_DESCRIPTIONS = {
        "intramolecular_passed": "The conformation of the molecule is reasonable with respect to Bond Lengths, Bond Angles, Internal Steric Clash, Aromatic Ring Flatness, Non-Aromatic Ring Non-Flatness, Double Bond Flatness",
        "intermolecular_passed": "The molecule interacts with the protein in a reasonable way with respect to Protein-Ligand Maximum Distance, Minimum Distance to Protein, Volume Overlap with Protein",
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        with open(Path(__file__).resolve().parents[2] / "configs" / "posebusters.yml", encoding="utf-8") as config_file:
            config = safe_load(config_file)
        self.pb_obj = PoseBusters(config=config)

    def compute(self, mols, configuration):
        protein_mol = Chem.MolFromPDBBlock(configuration.environment_pdb, removeHs=False)
        pb_results = self.pb_obj.bust(mols, mol_cond=protein_mol)
        pb_results["intramolecular_passed"] = pb_results[["bond_lengths", "bond_angles", "internal_steric_clash", "aromatic_ring_flatness", "non-aromatic_ring_non-flatness", "double_bond_flatness"]].all(axis=1)
        pb_results["intermolecular_passed"] = pb_results[["protein-ligand_maximum_distance", "minimum_distance_to_protein", "volume_overlap_with_protein"]].all(axis=1)
        return {
            "intramolecular_passed": pb_results["intramolecular_passed"].tolist(),
            "intermolecular_passed": pb_results["intermolecular_passed"].tolist(),
        }


class AnchorPointsReward(RewardFunction):
    """Anchor-point score with both 2D and 3D components.

    Reads ``anchor_points`` and ``seed_molecule`` from
    ``configuration`` at score time.  SMARTS construction and anchor-point
    matching happen per call (accepted cost for small batches).

    - **3D component**: A point is matched when at least one atom of the
      same element lies within the point radius.
    - **2D component**: Enabled anchor points are mapped to atoms in the
      seed molecule and converted to a SMARTS fragment.  Candidate
      molecules are scored by substructure match against this query.
    """

    name = "anchor_points"
    SCORE_DESCRIPTIONS = {
        "2d_passed": "Binary substructure match against SMARTS built from seed-molecule anchor atoms (0 or 1)",
        "3d_passed": "Fraction of anchor points with a same-element atom within radius in 3D (0 to 1)",
        "3d_atom_frac": "Fraction of atoms in the seed molecule that are within radius of an anchor point",
    }

    def compute(self, mols, configuration):
        if isinstance(configuration, list):
            return self._merge_per_mol_results(
                [self.compute([m], c) for m, c in zip(mols, configuration)]
            )
        anchor_points = configuration.anchor_points
        if not anchor_points:
            return {
                "2d_passed": [None] * len(mols),
                "3d_passed": [None] * len(mols),
                "3d_atom_frac": [None] * len(mols),
            }

        anchor_points_smarts = build_anchor_points_smarts(configuration.seed_molecule,
                                                          anchor_points)
        query = Chem.MolFromSmarts(anchor_points_smarts) if anchor_points_smarts else None

        passed_2d = []
        passed_3d = []
        atom_frac_3d = []
        for mol in mols:
            score_2d = None
            if mol is not None and query is not None:
                score_2d = True if mol.HasSubstructMatch(query) else False

            score_3d = None
            hit_count = 0
            if mol is not None and mol.GetNumConformers() > 0:
                coords = mol.GetConformer().GetPositions()
                symbols = [a.GetSymbol() for a in mol.GetAtoms()]
                for p in anchor_points:
                    p_name = p.get("name")
                    p_radius = float(p.get("radius", 0.5))
                    p_xyz = np.array([p["x"], p["y"], p["z"]], dtype=float)
                    matched = False
                    for i, atom_xyz in enumerate(coords):
                        if p_name and symbols[i] != p_name:
                            continue
                        if np.linalg.norm(atom_xyz - p_xyz) <= p_radius:
                            matched = True
                            break
                    if matched:
                        hit_count += 1
                score_3d = hit_count == len(anchor_points)
                atom_frac_3d.append(hit_count / len(anchor_points))
            passed_2d.append(score_2d)
            passed_3d.append(score_3d)

        return {
            "2d_passed": passed_2d,
            "3d_passed": passed_3d,
            "3d_atom_frac": atom_frac_3d,
        }


class PharmacophoreReward(RewardFunction):
    """Pharmacophore score.

    Reads ``pharmacophore_points`` from ``configuration``. A pose passes
    when every point has a distinct ligand feature of the same type within
    that point's ``radius`` (Angstroms), measured in the pose frame. The
    match is not alignment-invariant: moving the conformer outside the
    radii fails the filter.
    """

    name = "pharmacophore"
    SCORE_DESCRIPTIONS = {
        "passed": "Whether every pharmacophore point has a same-type feature within its radius (0 or 1)",
        "rms": "Unaligned RMSD (A) between matched feature centers and the pharmacophore points",
    }

    def compute(self, mols, configuration):
        if isinstance(configuration, list):
            return self._merge_per_mol_results(
                [self.compute([m], c) for m, c in zip(mols, configuration)]
            )
        pharmacophore_points = configuration.pharmacophore_points
        output = {"passed": [], "rms": []}
        for mol in mols:
            passed, rms = score_pharmacophore_points(mol, pharmacophore_points)
            output["passed"].append(passed)
            output["rms"].append(rms)
        return output


class PLIPointsReward(RewardFunction):
    """Pocket-ligand interaction score.
    
    Reads ``environment_pdb`` from ``configuration`` and builds the
    pocket and ligand objects per call (accepted cost for small batches).
    """

    name = "pocket_ligand_interaction"
    SCORE_DESCRIPTIONS = {
        "passed": "interactions match the required interactions",
    }

    def compute(self, mols, configuration):
        if isinstance(configuration, list):
            return self._merge_per_mol_results(
                [self.compute([m], c) for m, c in zip(mols, configuration)]
            )
        
        environment_pdb = configuration.environment_pdb
        pdb_mol = Chem.MolFromPDBBlock(environment_pdb, removeHs=False)
        if pdb_mol is None:
            return {
                "passed": [None] * len(mols)
            }
        prot = plf.Molecule(pdb_mol)
        output = {"passed": []}
        for mol in mols:
            interaction_list = get_ligand_pli_points(mol, prot)
            if configuration.pocket_ligand_interaction_points is not None:
                conf_set = set(
                    [p["name"]+p["residue"] for \
                        p in configuration.pocket_ligand_interaction_points]
                )
                lig_set = set(
                    [p["name"]+p["residue"] for p in interaction_list]
                )
                if conf_set.issubset(lig_set):
                    output["passed"].append(True)
                else:
                    output["passed"].append(False)
            else:
                output["passed"].append(None)

        return output


class UniDockReward(RewardFunction):
    """UniDock GPU-accelerated docking score.

    Runs UniDock and returns binding affinity in kcal/mol (negative =
    better binding).

    Per-configuration context (from ``configuration``):

    - ``environment_pdb``: **Full receptor PDB text** (not a path).
    - ``seed_molecule``: Reference ligand whose centroid defines the
      docking box center.

    Static parameters (from spec dict, set once in ``__init__``):

    - ``search_mode``: ``"ultra_fast"``, ``"very_fast"``, ``"fast"``
      (default), ``"balance"``, or ``"detail"``
    - ``device_id``: GPU device ID (default 0)
    - ``score_only``: If ``True``, score existing poses without local
      search (default ``False``)
    - ``box_size``: Docking box side length in Angstroms (default 30)
    - ``scoring_function``: UniDock scoring function (default ``"vina"``)
    - ``refine_step``: Refinement steps (default 5)
    - ``docking_seed``: Random seed (default 42)
    """

    name = "unidock"
    SCORE_DESCRIPTIONS = {
        "passed": "Unidock docking affinity the default threshold is -4 kcal/mol",
        "score": "UniDock docking affinity in kcal/mol (more negative = stronger binding)",
        "ROMol": "UniDock docking output molecule",
    }

    SEARCH_MODES = {
        "ultra_fast": (16, 5),
        "very_fast": (64, 10),
        "fast": (128, 20),
        "balance": (384, 40),
        "detail": (512, 40),
    }

    def __init__(
        self,
        search_mode="fast",
        device_id=0,
        score_only=False,
        box_size=30,
        docking_seed=42,
        scoring_function="vina",
        refine_step=5,
        passed_threshold=-6.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.search_mode = search_mode
        self.device_id = device_id
        self.score_only = score_only
        self.box_size = box_size
        self.docking_seed = docking_seed
        self.scoring_function = scoring_function
        self.refine_step = refine_step
        self.passed_threshold = passed_threshold
        self._unidock_executable = resolve_unidock_executable()

    def _run_unidock(
        self,
        mols: List[Chem.Mol],
        receptor_pdb: str,
        box_center,
    ) -> Tuple[List[Optional[float]], List[Optional[Chem.Mol]]]:
        """Run UniDock for *mols* sharing one receptor and one search box center."""
        if self.search_mode not in self.SEARCH_MODES:
            raise ValueError(
                f"Unknown search_mode '{self.search_mode}'. "
                f"Available: {list(self.SEARCH_MODES.keys())}"
            )

        exhaustiveness, max_step = self.SEARCH_MODES[self.search_mode]
        scores: List[Optional[float]] = [None] * len(mols)
        output_mols: List[Optional[Chem.Mol]] = [None] * len(mols)

        valid_indices: List[int] = []
        prepared_mols: List[Chem.Mol] = []
        for i, m in enumerate(mols):
            if m is None or m.GetNumConformers() == 0:
                continue
            try:
                prepared_mols.append(AllChem.AddHs(m, addCoords=True))
                valid_indices.append(i)
            except Exception:
                continue

        if not valid_indices:
            return scores, output_mols

        with tempfile.TemporaryDirectory(dir=_SHM_DIR) as temp_dir:
            pdbqt_str = pdb_to_pdbqt_cached(receptor_pdb)
            pdbqt_path = os.path.join(temp_dir, "receptor.pdbqt")
            with open(pdbqt_path, "w", encoding="utf-8") as fh:
                fh.write(pdbqt_str)

            ligand_files = []
            for idx, m_h in enumerate(prepared_mols):
                ligprep((m_h, f"ligand_{idx}"), temp_dir, save_format="sdf")
                ligand_files.append(
                    os.path.join(temp_dir, f"ligand_{idx}.sdf")
                )

            cmd = (
                f"{self._unidock_executable}"
                f" --receptor {pdbqt_path}"
                f" --gpu_batch {' '.join(ligand_files)}"
                f" {'--score_only' if self.score_only else '--local_only'}"
                f" --center_x {box_center[0]}"
                f" --center_y {box_center[1]}"
                f" --center_z {box_center[2]}"
                f" --size_x {self.box_size}"
                f" --size_y {self.box_size}"
                f" --size_z {self.box_size}"
                f" --verbosity 0"
                f" --device_id {self.device_id}"
                f" --dir {temp_dir}"
                f" --exhaustiveness {exhaustiveness}"
                f" --max_step {max_step}"
                f" --num_modes 1"
                f" --seed {self.docking_seed}"
                f" --scoring {self.scoring_function}"
                f" --refine_step {self.refine_step}"
            )

            ret = subprocess.run(cmd, shell=True, capture_output=True)

            if ret.returncode == 0:
                if self.score_only:
                    parsed = []
                    with open(os.path.join(temp_dir, "scores.txt")) as fh:
                        for line in fh:
                            if line.startswith("REMARK"):
                                parsed.append(float(line.split()[2]))
                    for orig_idx, s in zip(valid_indices, parsed):
                        scores[orig_idx] = s
                else:
                    for idx, orig_idx in enumerate(valid_indices):
                        out_file = os.path.join(
                            temp_dir, f"ligand_{idx}_out.sdf"
                        )
                        try:
                            for out_mol in Chem.SDMolSupplier(
                                out_file, removeHs=False
                            ):
                                if out_mol is not None:
                                    result_str = out_mol.GetPropsAsDict().get(
                                        "Uni-Dock RESULT", ""
                                    )
                                    scores[orig_idx] = float(
                                        result_str.split()[1]
                                    )
                                    output_mols[orig_idx] = out_mol
                        except Exception as exc:
                            logger.warning(
                                "Failed to parse UniDock output for "
                                "ligand %d: %s",
                                idx, exc,
                            )
            else:
                stdout = ret.stdout.decode("utf-8", errors="replace")
                stderr = ret.stderr.decode("utf-8", errors="replace")
                logger.error(
                    "UniDock failed (rc=%d):\ncmd: %s\n"
                    "stdout: %s\nstderr: %s",
                    ret.returncode, cmd, stdout, stderr,
                )
            return scores, output_mols

    def compute(self, mols, configuration):
        if isinstance(configuration, list):
            from concurrent.futures import ThreadPoolExecutor, as_completed

            scores = [None] * len(mols)
            output_mols = [None] * len(mols)
            with ThreadPoolExecutor(max_workers=min(len(mols), 64)) as pool:
                futures = {
                    pool.submit(self.compute, [m], c): i
                    for i, (m, c) in enumerate(zip(mols, configuration))
                }
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        results = future.result()
                        scores[idx] = results["score"][0]
                        output_mols[idx] = results["ROMol"][0]
                    except Exception:
                        logger.error(
                            "UniDockReward: parallel dock failed for mol %d:\n%s",
                            idx, traceback.format_exc(),
                        )
            return {
                "passed": [s < self.passed_threshold if s else False for s in scores],
                "score": scores,
                "ROMol": output_mols,
            }

        environment_pdb = configuration.environment_pdb
        if environment_pdb is None:
            logger.warning("UniDockReward: no environment_pdb in configuration")
            return {
                "passed": [None] * len(mols),
                "score": [None] * len(mols),
                "ROMol": [None] * len(mols),
            }

        seed_mol = configuration.seed_molecule
        if seed_mol is not None and seed_mol.GetNumConformers() > 0:
            box_center = seed_mol.GetConformer().GetPositions().mean(axis=0)
        else:
            # Fall back to the first molecule that actually has a conformer;
            # the batch may contain None / conformer-less entries.
            ref_mol = next(
                (m for m in mols if m is not None and m.GetNumConformers() > 0),
                None,
            )
            if ref_mol is None:
                logger.warning(
                    "UniDockReward: no seed molecule and no candidate with a "
                    "conformer to define the docking box center"
                )
                return {
                    "passed": [None] * len(mols),
                    "score": [None] * len(mols),
                    "ROMol": [None] * len(mols),
                }
            box_center = ref_mol.GetConformer().GetPositions().mean(axis=0)
        scores, output_mols = self._run_unidock(mols, environment_pdb, box_center)
        return {
            "passed": [s < self.passed_threshold if s else False for s in scores],
            "score": scores,
            "ROMol": output_mols,
        }


class EnergyReward(RewardFunction):
    """Per-bond MMFF energy."""

    name = "per_bond_mmff_energy"
    SCORE_DESCRIPTIONS = {
        "score": "MMFF94 force-field energy per bond (kcal/mol/bond)",
    }

    def compute(self, mols, configuration):
        energy_results = []
        for m in mols:
            try:
                props = AllChem.MMFFGetMoleculeProperties(m)
                ff = AllChem.MMFFGetMoleculeForceField(m, props)
                energy = ff.CalcEnergy() if ff else None
                energy_results.append(energy)
            except Exception:
                energy_results.append(None)

        energies = np.array(energy_results)
        num_bonds = np.array([m.GetNumBonds() for m in mols])
        energies = energies / (num_bonds + 1e-6)
        return {"score": energies.tolist()}


class LigandSimilarityReward(RewardFunction):
    """Shape/ESP similarity score.

    Reads ``seed_molecule`` from ``configuration`` as the reference.
    Compares each molecule's primary conformer (conformer 0) directly
    against the reference's primary conformer with no embedding or
    alignment -- the poses are evaluated exactly as-is.
    """

    name = "ligand_similarity"
    pass_threshold = 0.3
    SCORE_DESCRIPTIONS = {
        "fp": "Morgan fingerprint Tanimoto similarity to the reference ligand molecule (0 to 1)",
        "fp_passed": "Morgan fingerprint Tanimoto similarity is at least 0.3",
        "shape": "Shape (volume overlap) similarity to the reference ligand molecule (0 to 1)",
        "shape_passed": "Shape similarity is at least 0.3",
        "esp": "Electrostatic potential similarity to the reference ligand molecule (-1 to 1)",
        "esp_passed": "Electrostatic potential similarity is at least 0.3",
    }
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.fp_generator = AllChem.GetMorganGenerator(radius=2)

    def _passed(self, values):
        return [
            None if value is None else value >= self.pass_threshold
            for value in values
        ]

    def _empty_result(self, n):
        return {key: [None] * n for key in self.SCORE_DESCRIPTIONS}

    def compute(self, mols, configuration):
        if isinstance(configuration, list):
            return self._merge_per_mol_results(
                [self.compute([m], c) for m, c in zip(mols, configuration)]
            )
        reference_mol = configuration.seed_molecule
        if reference_mol is None:
            return self._empty_result(len(mols))

        sim_fp = []
        sim_shape = []
        sim_esp = []
        for mol in mols:
            if mol is None:
                sim_fp.append(None)
                sim_shape.append(None)
                sim_esp.append(None)
                continue
            try:
                fp1 = self.fp_generator.GetSparseCountFingerprint(mol)
                fp2 = self.fp_generator.GetSparseCountFingerprint(reference_mol)
                f = DataStructs.TanimotoSimilarity(fp1, fp2)
                s = GetShapeSim(mol, reference_mol, prbCid=0, refCid=0)
                e = GetEspSim(mol, reference_mol, prbCid=0, refCid=0)
                sim_fp.append(f)
                sim_shape.append(s)
                sim_esp.append(e)
            except Exception:
                logger.warning(
                    "LigandSimilarityReward: scoring failed for mol:\n%s",
                    traceback.format_exc(),
                )
                sim_fp.append(None)
                sim_shape.append(None)
                sim_esp.append(None)

        return {
            "fp": sim_fp,
            "fp_passed": self._passed(sim_fp),
            "shape": sim_shape,
            "shape_passed": self._passed(sim_shape),
            "esp": sim_esp,
            "esp_passed": self._passed(sim_esp),
        }

# ---------------------------------------------------------------------------
# Auto-register build-in reward functions
# ---------------------------------------------------------------------------
register_reward(DescriptorsReward)
register_reward(PoseBustersReward)
register_reward(UniDockReward)
register_reward(AnchorPointsReward)
register_reward(PharmacophoreReward)
register_reward(PLIPointsReward)
register_reward(LigandSimilarityReward)