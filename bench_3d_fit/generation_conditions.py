"""GenerationConditions dataclass and prompt construction."""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import json
from rdkit import Chem
import numpy as np
import pandas as pd

from bench_3d_fit.prompt import (
    format_environment_pdb,
    format_anchor_points,
    format_pharmacophore_points,
    format_pocket_ligand_interaction_points,
    format_output_instructions,
    base_molecule_conditions,
)
from bench_3d_fit.mol_io import mol_from_molblock_text

logger = logging.getLogger(__name__)

@dataclass
class GenerationConditions:
    """Conditions for a single molecular generation task.

    Encapsulates everything needed to condition a molecular generation
    and to score the generated molecules against the core reward functions.

    Use :meth:`to_prompt` to build an instruction-following LLM prompt, or
    :meth:`to_dict` to extract the raw condition fields for specialist models.

    Instances are created either by
    :func:`~bench_3d_fit.sampling.sample_generation_conditions`
    (sampling conditions from a test-set row at runtime) or by
    :meth:`from_test_row` (loading pre-sampled conditions stored in a parquet).
    """

    # Reference molecule and usual (though not required) source of 3D conditions
    seed_molecule: Optional[Chem.Mol] = None

    # Environment pdb of the protein POCKET only -
    # See process_raw_data.py for more details on pocket selection
    environment_pdb: Optional[str] = None

    # Anchor points for the generation.
    # Each entry is a dict with keys: 'name' (atomic symbol), 'x', 'y', 'z',
    # and 'radius' (match tolerance in Angstroms).
    # Example:
    # [
    #     {'name': 'C', 'x': -38.421, 'y': 9.296, 'z': 1.732, 'radius': 0.5},
    #     {'name': 'O', 'x': -36.941, 'y': 9.986, 'z': 1.868, 'radius': 0.5},
    # ]
    anchor_points: Optional[List[Dict[str, Any]]] = None

    # Pharmacophore points for the generation.
    # Each entry is a dict with keys: 'name' (pharmacophore type name),
    # 'x', 'y', 'z', and 'radius' (match tolerance in Angstroms).
    # Example:
    # [
    #     {'name': 'HBDonor', 'x': 0.0, 'y': 0.0, 'z': 0.0, 'radius': 0.5},
    #     {'name': 'HBAcceptor', 'x': 1.0, 'y': 0.0, 'z': 0.0, 'radius': 0.5},
    # ]
    # Pharmacophore type names:
    #   HBDonor, HBAcceptor, Hydrophobic, PositiveIonizable,
    #   NegativeIonizable, Aromatic, Exclusion
    pharmacophore_points: Optional[List[Dict[str, Any]]] = None

    # Pocket-ligand interaction points for the generation.
    # Each entry is a dict with keys: 'name' (interaction type) and
    # 'residue' (e.g. "TRP199.A").  Multiple interactions with the same
    # residue are listed separately.
    # Example:
    # [
    #   {'name': 'VdWContact', 'residue': 'TRP199.A'},
    #   {'name': 'Hydrophobic', 'residue': 'TRP199.A'},
    #   {'name': 'HBDonor', 'residue': 'ASP114.D'},
    # ]
    # Interaction types:
    #   Hydrophobic, HBDonor, HBAcceptor, PiStacking, Anionic,
    #   Cationic, CationPi, PiCation, VdWContact
    pocket_ligand_interaction_points: Optional[List[Dict[str, str]]] = None

    # ---------------------------------- #
    # Accessors for Molecular Generation #
    # to_prompt() for LLMs
    # to_dict() for other models
    # ---------------------------------- #
    def to_prompt(self,
        conditions_to_include: Optional[List[str]] = ['pocket', 'anchor_points', 'pharmacophore', 'pocket_ligand_interaction'],
        format_instructions: str = "sdf_enum",
        format_tag: str = "mol",
        mol_conditions_version: str = "druglike_ro5",
        pocket_n_atoms: int = 100,
    ) -> str:
        """Build a natural-language prompt for instruction-following LLMs.

        Args:
            conditions_to_include: Which condition blocks to include in the
                prompt.  Valid tokens: ``'pocket'``, ``'anchor_points'``,
                ``'pharmacophore'``, ``'pocket_ligand_interaction'``.
            format_instructions: Output format the LLM should use.  One of
                ``'smiles_xyz'``, ``'sdf'``, ``'sdf_enum'``,
                ``'smiles_enum'``, ``'smiles_edits'``.
            format_tag: XML-style tag that wraps the generated molecule.
            mol_conditions_version: Druglikeness prompt variant. One of
                ``'base'``, ``'druglike_nheavy_25'``, ``'druglike_nheavy_15'``,
                ``'druglike_nheavy_seed'``, ``'druglike_ro5'``.
            pocket_n_atoms: Maximum pocket atoms written into the prompt,
                keeping those closest to the seed ligand.

        Returns:
            A self-contained prompt string ready to send to the model.
        """
        mol_desc = base_molecule_conditions(version=mol_conditions_version, seed_molecule=self.seed_molecule)

        prompt = "Generate the three-dimensional structure of a ligand molecule given the following 3D generation conditions:\n"
        prompt += mol_desc
        if "pocket" in conditions_to_include and self.environment_pdb is not None:
            pdb_text = self.reduce_protein_pocket_atoms(n_atoms=pocket_n_atoms)
            prompt += f"The ligand molecule must bind to the following protein pocket, meaning good interactions with the protein pocket atoms and no clashes (atoms with very close coordinates) with the protein pocket atoms:\n"
            prompt += f"{format_environment_pdb(pdb_text)}\n"
        if "anchor_points" in conditions_to_include and self.anchor_points is not None:
            prompt += f"The ligand molecule must contain the following atoms and their 3D coordinates within 0.5 Angstroms of the given coordinates:\n"
            prompt += f"{format_anchor_points(self.anchor_points)}\n"
        if "pharmacophore" in conditions_to_include and self.pharmacophore_points is not None:
            prompt += f"The ligand molecule must have the following pharmacophore points:\n"
            prompt += f"{format_pharmacophore_points(self.pharmacophore_points)}\n"
        if "pocket_ligand_interaction" in conditions_to_include and self.pocket_ligand_interaction_points is not None:
            prompt += f"The ligand molecule must have the following interactions with specific residues in the protein pocket:\n"
            prompt += f"{format_pocket_ligand_interaction_points(self.pocket_ligand_interaction_points)}\n"
        prompt += "The most important condition is generating a valid molecule with plausible 2D and 3D structure. This means reasonable bond lengths, angles, and no overlapping with protein pocket.\n"
        prompt += "If all conditions are not satifisable simulatenously, prioritize the validity of the generated ligand molecule.\n"
        prompt += f"The output molecule must have a valid 2D and 3D structure enclosed in <{format_tag}> and </{format_tag}> tags.\n"
        prompt += format_output_instructions(format_instructions, format_tag)
        prompt += "Important: Before finalizing, explicitly verify, to the best of your ability, the output molecule: format validity, 3d plausibility, all heavy atoms have associated coordinates, and all conditions are satisfied.\n"
        prompt += "If uncertain or unable to satisfy all conditions, prefer a ligand molecule that satistifes the original validity conditions, however a molecule MUST be generated.\n"
        prompt += "If information you deem required is missing, make predictions based on your knowledge and the context provided:\n"
        prompt += mol_desc
        prompt += f"If you see the problems with unrealistic atom coordinates, stop generating the current molecule, close the tags and generate a new molecule from scratch in a new set of <{format_tag}> and </{format_tag}> tags.\n"
        prompt += "Repeat this process as necessary before outputting the final answer. A molecule MUST be generated to the best of your ability to satisfy all conditions."
        return prompt

    def reduce_protein_pocket_atoms(self, n_atoms: int = 1000) -> str:
        """Reduce the protein pocket atoms to a smaller subset of atoms."""
        if self.environment_pdb is None or self.seed_molecule is None:
            return self.environment_pdb

        pocket_atom_lines = np.array([l for l in self.environment_pdb.split("\n") if l.startswith("ATOM")])
        if len(pocket_atom_lines) <= n_atoms:
            return self.environment_pdb

        pocket_coords = np.array([[float(l[30:38].strip()), float(l[38:46].strip()), float(l[46:54].strip())] for l in pocket_atom_lines])
        reference_coords = np.array(self.seed_molecule.GetConformer().GetPositions())
        min_distances = np.linalg.norm(
            pocket_coords[:, None, :] - reference_coords[None, :, :], axis=2
        ).min(axis=1)
        keep_idxs = np.argsort(min_distances)[:n_atoms].tolist()
        return "\n".join(pocket_atom_lines[keep_idxs].tolist())

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain dict of the condition fields for non-LLM generators."""

        conditions: Dict[str, Any] = {
            "seed_molecule": self.seed_molecule,
            "environment_pdb": self.environment_pdb,
            "anchor_points": self.anchor_points,
            "pharmacophore_points": self.pharmacophore_points,
            "pocket_ligand_interaction_points": self.pocket_ligand_interaction_points,
        }
        return conditions

    @classmethod
    def from_test_row(cls, test_row: pd.Series) -> "GenerationConditions":
        """Build from a test-set row with pre-sampled condition columns.

        Expects the row to contain ``sdf``, ``pdb``, ``anchor_points``,
        ``pharmacophore_points``, and ``pli_points`` (the latter three as
        JSON strings produced by the ``sample_dataset_conditions`` script).
        """
        return GenerationConditions(
            seed_molecule=Chem.RemoveHs(mol_from_molblock_text(test_row["sdf"])),
            environment_pdb=test_row["pdb"],
            anchor_points=json.loads(test_row["anchor_points"]),
            pharmacophore_points=json.loads(test_row["pharmacophore_points"]),
            pocket_ligand_interaction_points=json.loads(test_row["pli_points"])
        )
