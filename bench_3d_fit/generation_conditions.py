"""GenerationConditions dataclass and prompt construction."""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import json
from rdkit import Chem
import pandas as pd

from bench_3d_fit.prompt import (
    format_environment_pdb,
    format_anchor_points,
    format_pharmacophore_points,
    format_pocket_ligand_interaction_points,
    format_output_instructions,
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
        format_instructions: str = "sdf_enum"
    ) -> str:
        """Build a natural-language prompt for instruction-following LLMs.

        Args:
            conditions_to_include: Which condition blocks to include in the
                prompt.  Valid tokens: ``'pocket'``, ``'anchor_points'``,
                ``'pharmacophore'``, ``'pocket_ligand_interaction'``.
            format_instructions: Output format the LLM should use.  One of
                ``'smiles_xyz'``, ``'sdf'``, ``'sdf_enum'``,
                ``'smiles_enum'``, ``'smiles_edits'``.

        Returns:
            A self-contained prompt string ready to send to the model.
        """

        prompt = "Generate the three-dimensional structure of a ligand molecule given the following 3D generation conditions:\n"
        prompt += f"The ligand molecule must be a valid molecule, meaning it is a single connected structure with correct valencies for all atoms.\n"
        prompt += f"The ligand molecule must have a 3D structure, and the bond lengths and angles must obey chemical rules such that the strain energy of the molecule is minimal.\n"
        if "pocket" in conditions_to_include and self.environment_pdb is not None:
            prompt += f"The ligand molecule must bind to the following protein pocket, meaning good interactions with the protein pocket atoms and no clashes (atoms with very close coordinates) with the protein pocket atoms:\n"
            prompt += f"{format_environment_pdb(self.environment_pdb)}\n"
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
        format_tag = "sdf" if format_instructions in ["sdf", "sdf_enum"] else "smiles_xyz"
        prompt += f"The output molecule must have a valid 2D and 3D structure enclosed in <{format_tag}> and </{format_tag}> tags.\n"
        prompt += format_output_instructions(format_instructions)
        prompt += "Important: Before finalizing, explicitly verify the validity of answer: format validity, SMILES validity, all heavy atoms in SMILES has at least one entry with coordinates, the final 3d structure should be physically plausible, there shouldn't be any overlaps with protein pocket.\n"
        prompt += "If uncertain, prefer a molecular ligand output validity over satisfying other constraints. Do not output any molecule unless all checks pass. If you see the problems with unrealistic atom coordinates, "
        if format_instructions == "smiles_edits":
            prompt += "add additional coordinates entries to overwrite the wrong coordinates. If impossible to fix the output with coordinates editing, "
        prompt += f"stop generating the current molecule, close the tags and generate a new molecule from scratch in a new set of <{format_tag}> and </{format_tag}> tags. Repeat this process as necessary to output a valid molecule."
        return prompt

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
            seed_molecule=mol_from_molblock_text(test_row["sdf"]),
            environment_pdb=test_row["pdb"],
            anchor_points=json.loads(test_row["anchor_points"]),
            pharmacophore_points=json.loads(test_row["pharmacophore_points"]),
            pocket_ligand_interaction_points=json.loads(test_row["pli_points"])
        )
