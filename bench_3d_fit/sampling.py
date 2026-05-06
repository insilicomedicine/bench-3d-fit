#!/usr/bin/env python3
"""
Sample :class:`~bench_3d_fit.generation_conditions.GenerationConditions`
from a test-set row (in-memory; no temp files).

The input row must contain at least ``sdf`` (MolBlock text) and ``pdb``
(protein pocket PDB text).  Anchor points are sampled from BRICS fragments,
pharmacophore points from the ligand's pharmacophore features, and
pocket-ligand interaction points from the ProLIF fingerprint.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Sequence, Tuple
import pandas as pd
from rdkit import Chem
from rdkit.Chem import BRICS

from bench_3d_fit.mol_io import (
    mol_from_molblock_text,
    require_parquet_cell_text,
)
from bench_3d_fit.generation_conditions import GenerationConditions
from bench_3d_fit.reward.utils import (
    get_ligand_pharmacophore_points,
    get_ligand_pli_points,
)


def sample_generation_conditions(
    row: pd.Series,
    min_ph4_features: int = 1,
    max_ph4_features: int = 1,
    pharmacophore_radius: float = 0.5,
    preferred_min_frag_size: int = 3,
    preferred_max_frag_size: int = 8,
    anchor_radius: float = 0.5,
    min_pli_points: int = 1,
    max_pli_points: int = 1,
    config_sampling_seed: int = 42,
) -> GenerationConditions:
    """Sample a complete :class:`GenerationConditions` from a test-set row.

    Extracts the ligand and pocket from the row, fragments the ligand via
    BRICS, and randomly samples anchor points, pharmacophore features, and
    pocket-ligand interaction points to form the generation conditions.

    Args:
        row: A pandas Series (or dict-like) with at least ``'sdf'`` and
            ``'pdb'`` keys containing MolBlock and PDB text respectively.
        min_ph4_features: Minimum pharmacophore features to sample.
        max_ph4_features: Maximum pharmacophore features to sample.
        pharmacophore_radius: Tolerance radius for pharmacophore points (A).
        preferred_min_frag_size: Preferred minimum heavy-atom count for
            the sampled BRICS fragment.
        preferred_max_frag_size: Preferred maximum heavy-atom count for
            the sampled BRICS fragment.
        anchor_radius: Tolerance radius for anchor points (A).
        min_pli_points: Minimum pocket-ligand interaction points to sample.
        max_pli_points: Maximum pocket-ligand interaction points to sample.
        config_sampling_seed: RNG seed for reproducible sampling.

    Returns:
        A fully populated :class:`GenerationConditions` instance.
    """
    
    rng = random.Random(config_sampling_seed)
    sdf_text = require_parquet_cell_text(row["sdf"], "sdf")
    pdb_text = require_parquet_cell_text(row["pdb"], "pdb")

    ligand = mol_from_molblock_text(sdf_text)
    atom_idxs = _pick_random_brics_fragment_atom_indices(
        ligand,
        preferred_min_frag_size=preferred_min_frag_size,
        preferred_max_frag_size=preferred_max_frag_size,
        rng=rng,
    )
    anchor_points = _anchor_points_from_fragment(ligand, atom_idxs, radius=anchor_radius)

    all_ligand_pharmacophore_points = get_ligand_pharmacophore_points(
        ligand,
        radius=pharmacophore_radius
    )
    pharmacophore_points = _sample_points(
        all_ligand_pharmacophore_points,
        min_points=min_ph4_features,
        max_points=max_ph4_features,
        rng=rng,
    )
    all_pocket_ligand_interaction_points = get_ligand_pli_points(ligand, pdb_text)
    pocket_ligand_interaction_points = _sample_points(
        all_pocket_ligand_interaction_points,
        rng=rng,
        min_points=min_pli_points,
        max_points=max_pli_points,
    )
    return GenerationConditions(
        seed_molecule=ligand,
        environment_pdb=pdb_text,
        anchor_points=anchor_points,
        pharmacophore_points=pharmacophore_points,
        pocket_ligand_interaction_points=pocket_ligand_interaction_points
    )


# ------------------------------------------------------------------------- #
#                Helper functions for condition sampling                    #
# ------------------------------------------------------------------------- #


def _fragment_molecule_brics(mol: Chem.Mol) -> Tuple[Tuple[int, ...], ...]:
    brics_bonds = list(BRICS.FindBRICSBonds(mol))
    if not brics_bonds:
        return [[a.GetIdx()] for a in mol.GetAtoms() if a.GetAtomicNum() > 1]

    bond_ids: List[int] = []
    for (i, j), _labels in brics_bonds:
        b = mol.GetBondBetweenAtoms(i, j)
        if b is not None:
            bond_ids.append(b.GetIdx())
    if not bond_ids:
        return [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]

    frag_mol = Chem.FragmentOnBonds(mol, bond_ids, addDummies=False)
    frags = Chem.GetMolFrags(frag_mol, asMols=False, sanitizeFrags=False)
    if not frags:
        return [[a.GetIdx()] for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    return frags


def _fragment_molecule_rotatable_bonds(mol: Chem.Mol) -> Tuple[Tuple[int, ...], ...]:
    pattern = Chem.MolFromSmarts("[*]!@-[*]")
    bonds_to_break = mol.GetSubstructMatches(pattern)
    
    bond_ids = [mol.GetBondBetweenAtoms(a1, a2).GetIdx() for a1, a2 in bonds_to_break]
    
    frag_mol = Chem.FragmentOnBonds(mol, bond_ids, addDummies=False)
    frags = Chem.GetMolFrags(frag_mol, asMols=False, sanitizeFrags=False)
    if not frags:
        return [[a.GetIdx()] for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    return frags


def _pick_random_brics_fragment_atom_indices(
    mol: Chem.Mol, preferred_min_frag_size: int = 3,
    preferred_max_frag_size: int = 8, rng: random.Random = None,
) -> List[int]:
    if rng is None:
        rng = random.Random()
    brics_frags = _fragment_molecule_brics(mol)
    good_brics_frags = [f for f in brics_frags if len(f) >= preferred_min_frag_size and len(f) <= preferred_max_frag_size]
    if not good_brics_frags:
        rot_bond_frags = _fragment_molecule_rotatable_bonds(mol)
        good_rot_bond_frags = [f for f in rot_bond_frags if len(f) >= preferred_min_frag_size and len(f) <= preferred_max_frag_size]
        if not good_rot_bond_frags:
            chosen = rng.choice(brics_frags)
        else:
            chosen = rng.choice(good_rot_bond_frags)
    else:
        chosen = rng.choice(good_brics_frags)
    atom_idxs = []
    for idx in chosen:
        a = mol.GetAtomWithIdx(idx)
        if a.GetAtomicNum() > 1:
            atom_idxs.append(idx)
    if not atom_idxs:
        raise ValueError("No fragment atom indices found in molecule")
    return atom_idxs


def _anchor_points_from_fragment(
    mol: Chem.Mol, atom_idxs: Sequence[int], radius: float = 0.5
) -> List[Dict[str, Any]]:
    conf = mol.GetConformer()
    points: List[Dict[str, Any]] = []
    for idx in atom_idxs:
        pos = conf.GetAtomPosition(idx)
        atom = mol.GetAtomWithIdx(idx)
        points.append(
            {
                "name": atom.GetSymbol(),
                "x": float(pos.x),
                "y": float(pos.y),
                "z": float(pos.z),
                "radius": radius,
            }
        )
    return points


def _sample_points(
    all_points: List[Any],
    min_points: int,
    max_points: int,
    rng: random.Random = None,
) -> List[Any]:
    if rng is None:
        rng = random.Random()
    if len(all_points) < min_points:
        min_points = len(all_points)
    k = min(
        len(all_points),
        rng.randint(max(1, min_points), max(min_points, max_points))
    )
    return rng.sample(all_points, k=k)
