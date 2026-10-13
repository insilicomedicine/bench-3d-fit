"""
Helpers for reward scoring that are not tied to a single ``RewardFunction``.

Receptor handling for UniDock uses :func:`pdb_to_pdbqt` (ProDy + RDKit + UniDock
tools) so ``rewards.py`` stays focused on reward classes.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import threading
from collections import defaultdict
from io import StringIO

import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import Any, Dict, List, Optional, Tuple, Union
from pmapper.pharmacophore import Pharmacophore as PH4

import prody
from rdkit import Chem
from rdkit.Chem import SanitizeFlags
from unidock_tools.modules.protein_prep.pdb2pdbqt import receptor_mol_to_pdbqt_str
import prolif as plf

logger = logging.getLogger(__name__)


def resolve_unidock_executable() -> str:
    """
    Locate the UniDock binary: :func:`shutil.which` for the ``unidock`` command
    on ``PATH``, otherwise the ``UNIDOCK_BIN`` environment variable.

    Raises:
        ValueError: If ``unidock`` is not on ``PATH`` and ``UNIDOCK_BIN`` is
            unset, or if ``UNIDOCK_BIN`` is set but is not an executable file.
    """
    path_exe = shutil.which("unidock")
    if path_exe:
        return path_exe
    env = os.environ.get("UNIDOCK_BIN")
    if not env:
        raise ValueError(
            "UniDock executable not found on PATH (command 'unidock') and environment "
            "variable UNIDOCK_BIN is not set. Unidock isn't installed in the environment "
            "during pip installation and needs to be installed separately. "
            "See the README.md for instructions on installing UniDock."
        )
    if not (os.path.isfile(env) and os.access(env, os.X_OK)):
        raise ValueError(
            f"UNIDOCK_BIN is set to {env!r} but it is not a valid executable file."
        )
    return os.path.abspath(env)


def pdb_to_pdbqt(pdb_str: str) -> str:
    """
    Convert a PDB string to a PDBQT string using ProDy, RDKit and UniDock tools.
    """
    if not pdb_str.strip():
        raise ValueError("PDB input is empty.")

    res = prody.parsePDBStream(StringIO(pdb_str))
    if res is None:
        raise ValueError("ProDy could not parse PDB content (invalid or empty).")

    out_stream = StringIO()
    prody.writePDBStream(out_stream, res)
    pocket_pdb_str = out_stream.getvalue()

    protein_mol = Chem.MolFromPDBBlock(pocket_pdb_str, removeHs=False)
    if protein_mol is None:
        protein_mol = Chem.MolFromPDBBlock(
            pocket_pdb_str, sanitize=False, removeHs=False
        )
        if protein_mol is None:
            raise ValueError("RDKit could not parse pocket PDB block from ProDy.")
        Chem.SanitizeMol(
            protein_mol,
            SanitizeFlags.SANITIZE_ALL ^ SanitizeFlags.SANITIZE_PROPERTIES,
        )

    return receptor_mol_to_pdbqt_str(protein_mol)


_pdbqt_cache: Dict[str, str] = {}
_pdbqt_cache_lock = threading.Lock()


def pdb_to_pdbqt_cached(pdb_str: str) -> str:
    """Like :func:`pdb_to_pdbqt` but caches results keyed by PDB content hash.

    Thread-safe.  Useful when multiple molecules share the same receptor
    across batches or parallel workers.
    """
    key = hashlib.sha256(pdb_str.encode("utf-8")).hexdigest()
    with _pdbqt_cache_lock:
        cached = _pdbqt_cache.get(key)
    if cached is not None:
        logger.debug("pdb_to_pdbqt cache hit (%s…)", key[:12])
        return cached
    result = pdb_to_pdbqt(pdb_str)
    with _pdbqt_cache_lock:
        _pdbqt_cache[key] = result
    return result


def build_anchor_points_smarts(seed_molecule: Chem.Mol, anchor_points: list) -> Optional[str]:
    """Build a SMARTS pattern from the seed molecule atoms closest to each anchor point."""
    if not anchor_points or seed_molecule is None:
        return None
    if seed_molecule.GetNumConformers() == 0:
        logger.warning(
            "AnchorPointsReward: seed_molecule has no conformer; cannot build 2D SMARTS from anchor points"
        )
        return None

    coords = seed_molecule.GetConformer().GetPositions()
    symbols = [a.GetSymbol() for a in seed_molecule.GetAtoms()]
    matched_indices = []
    for p in anchor_points:
        p_name = p.get("name")
        p_xyz = np.array([p["x"], p["y"], p["z"]], dtype=float)

        best_idx = None
        best_dist = None
        for i, atom_xyz in enumerate(coords):
            if p_name and symbols[i] != p_name:
                continue
            dist = np.linalg.norm(atom_xyz - p_xyz)
            if best_dist is None or dist < best_dist:
                best_idx = i
                best_dist = dist
        if best_idx is not None:
            matched_indices.append(best_idx)

    unique_indices = sorted(set(matched_indices))
    if not unique_indices:
        logger.warning("AnchorPointsReward: no anchor points map to seed_molecule atoms")
        return None

    try:
        return Chem.MolFragmentToSmarts(seed_molecule, atomsToUse=unique_indices)
    except Exception as exc:
        logger.warning("AnchorPointsReward: failed to build SMARTS fragment: %s", exc)
        return None


# pmapper feature codes. ``radius`` on a stored point is a Euclidean match
# tolerance in Angstroms; it is not a pmapper ``bin_step``.
_PH4_CODE_TO_NAME = {
    "D": "HBDonor",
    "A": "HBAcceptor",
    "H": "Hydrophobic",
    "P": "PositiveIonizable",
    "N": "NegativeIonizable",
    "a": "Aromatic",
    "e": "Exclusion",
}
# Cost used for feature/point pairs outside the point radius so the
# assignment prefers a feasible one-to-one match when one exists.
_PH4_INFEASIBLE_COST = 1e12


def get_ligand_pharmacophore_points(ligand: Chem.Mol, radius: float = 0.5) -> List[Dict[str, Any]]:
    """Extract pharmacophore feature points from a ligand molecule.

    ``radius`` is stored on each point as the spatial match tolerance (A).
    It does not change which features are returned.
    """
    q = PH4()
    q.load_from_mol(ligand)
    feature_coords = q.get_feature_coords()
    points = []
    for feature_coord in feature_coords:
        points.append({
            'name': _PH4_CODE_TO_NAME[feature_coord[0]],
            'x': feature_coord[1][0],
            'y': feature_coord[1][1],
            'z': feature_coord[1][2],
            'radius': radius
        })
    return points


def _pharmacophore_features_by_name(mol: Chem.Mol) -> Dict[str, np.ndarray]:
    """Ligand feature centers in the pose frame, keyed by type name."""
    q = PH4()
    q.load_from_mol(mol)
    grouped: Dict[str, List[Tuple[float, float, float]]] = defaultdict(list)
    for code, xyz in q.get_feature_coords():
        name = _PH4_CODE_TO_NAME.get(code)
        if name is None:
            continue
        grouped[name].append(xyz)
    return {name: np.asarray(coords, dtype=float) for name, coords in grouped.items()}


def score_pharmacophore_points(
    mol: Optional[Chem.Mol],
    pharmacophore_points: Optional[List[Dict[str, Any]]],
    default_radius: float = 0.5,
) -> Tuple[Optional[bool], Optional[float]]:
    """Match hypothesis points to ligand features in the pose coordinate frame.

    Each point passes when a distinct feature of the same type lies within
    that point's ``radius`` (A). No superposition is applied: a rigid
    translation that moves features outside their radii fails. Returns
    ``(passed, rms)``. ``rms`` is the unaligned RMSD of the minimum-cost
    feasible assignment, and is ``None`` when the hypothesis does not match
    or the molecule has no coordinates.
    """
    if mol is None or mol.GetNumConformers() == 0 or not pharmacophore_points:
        return None, None

    features = _pharmacophore_features_by_name(mol)
    by_name: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for point in pharmacophore_points:
        by_name[point["name"]].append(point)

    chosen_sq: List[float] = []
    for name, group in by_name.items():
        coords = features.get(name)
        if coords is None or len(coords) < len(group):
            return False, None
        targets = np.array(
            [[p["x"], p["y"], p["z"]] for p in group], dtype=float
        )
        radii = np.array(
            [float(p.get("radius", default_radius)) for p in group], dtype=float
        )
        dist = np.linalg.norm(targets[:, None, :] - coords[None, :, :], axis=2)
        feasible = dist <= radii[:, None]
        if not feasible.any(axis=1).all():
            return False, None
        cost = np.where(feasible, dist ** 2, _PH4_INFEASIBLE_COST)
        rows, cols = linear_sum_assignment(cost)
        picked = dist[rows, cols]
        if np.any(picked > radii[rows]):
            return False, None
        chosen_sq.extend(float(d) ** 2 for d in picked)
    rms = float(np.sqrt(np.mean(chosen_sq)))
    return True, rms


def get_ligand_pli_points(mol: Chem.Mol, prot: Union[plf.Molecule, str]) -> List[Dict[str, str]]:
    """Extract pocket-ligand interaction points using ProLIF fingerprints."""
    PLI_INTERACTION_NAMES = [
        'Hydrophobic', 'HBDonor', 'HBAcceptor', 'PiStacking', 'Anionic',
        'Cationic', 'CationPi', 'PiCation', 'VdWContact'
    ]
    if isinstance(prot, str):
        try:
            prot = plf.Molecule(Chem.MolFromPDBBlock(prot, removeHs=False))
        except Exception as exc:
            logger.warning("Prolif Pfailed to parse PDB block: %s", exc)
            return []
    lig = plf.Molecule(mol)
    fp = plf.Fingerprint()
    try:
        interactions = fp.generate(lig, prot)
    except Exception as exc:
        logger.warning("Prolif Fingerprint failed to generate interactions: %s", exc)
        interactions = {}
    interaction_list = []
    for point in interactions.keys():
        for interaction in [PLI_INTERACTION_NAMES[i] for i in range(len(PLI_INTERACTION_NAMES)) if interactions[point][i]]:
            interaction_list.append(
                {
                    "name": interaction,
                    "residue": str(point[1]),
                }
            )
    return interaction_list