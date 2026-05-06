"""Molecule I/O: parsing, reconstruction, and coordinate utilities.

Handles conversion between parquet table cells (SDF text, bytes, NaN) and
RDKit ``Mol`` objects.  Also provides reconstruction functions for the
various LLM output formats (SMILES+XYZ, simplified SDF, atom-mapped SMILES)
and coordinate transformation helpers.
"""

from __future__ import annotations
from typing import Any, Optional

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.rdMolTransforms import CanonicalizeMol
from rdkit.Geometry import Point3D
from scipy.stats import special_ortho_group


def coerce_parquet_cell_text(value: Any) -> Optional[str]:
    """Return normalized text, or None if the cell is missing or empty."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, memoryview):
        value = value.tobytes()
    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="replace")
    else:
        text = str(value)
    text = text.replace("\r\n", "\n")
    if not text.strip():
        return None
    return text


def require_parquet_cell_text(value: Any, field: str) -> str:
    """Like :func:`coerce_parquet_cell_text` but raises if missing or empty."""
    text = coerce_parquet_cell_text(value)
    if text is None:
        raise ValueError(f"Missing required field: {field}")
    return text


def mol_from_molblock_text(text: str) -> Chem.Mol:
    """Parse a single MolBlock (must have a conformer). Used after text is normalized."""
    mol = Chem.MolFromMolBlock(text, sanitize=True, removeHs=False)
    if mol is None:
        raise ValueError("Could not parse sdf text into RDKit Mol")
    if mol.GetNumConformers() == 0:
        raise ValueError("Ligand has no conformer in sdf text")
    if mol.GetNumConformers() > 1:
        raise ValueError("Ligand has multiple conformers in sdf text - Should have 1")
    return mol


def mol_from_sdf_cell(value: Any) -> Optional[Chem.Mol]:
    """
    Lenient ligand from a table cell: missing/empty → None; tries first record,
    then ``$$$$``-split records (annotation paths).
    """
    text = coerce_parquet_cell_text(value)
    if text is None:
        return None

    mol = Chem.MolFromMolBlock(text, sanitize=True, removeHs=False)
    if mol is not None:
        return mol

    for chunk in (c for c in text.split("$$$$") if c.strip()):
        m = Chem.MolFromMolBlock(chunk, sanitize=True, removeHs=False)
        if m is not None:
            return m
    return None


def safe_mol_to_mol_block(mol: Optional[Chem.Mol]) -> str:
    """Convert an RDKit Mol to a MolBlock string, returning ``""`` on failure."""
    if mol is None:
        return ""
    try:
        return Chem.MolToMolBlock(mol)
    except Exception as e:
        print(f"Error converting ROMol to MolBlock: {e}")
        return ""


def safe_mol_to_smiles(mol: Optional[Chem.Mol]) -> str:
    """Convert an RDKit Mol to a SMILES string, returning ``""`` on failure."""
    if mol is None:
        return ""
    try:
        return Chem.MolToSmiles(mol)
    except Exception as e:
        print(f"Error converting ROMol to SMILES: {e}")
        return ""


def reconstruct_smiles_xyz(s: str, canonicalize: bool = False) -> Optional[Chem.Mol]:
    """Reconstruct an RDKit Mol from ``<smiles_xyz>`` formatted text."""
    s = s.replace('<smiles_xyz>', '').replace('</smiles_xyz>', '')
    try:
        lines = s.split("|")
        smiles, coords_lines = lines[0], lines[1:]
        mol = Chem.MolFromSmiles(smiles)
        conf = Chem.Conformer()
        coords_lines = [line for line in coords_lines if "H" not in line]
        for i, line in enumerate(coords_lines):
            atom, x, y, z = line.split()
            conf.SetAtomPosition(i, [float(x), float(y), float(z)])

        mol.AddConformer(conf)

        if canonicalize:
            CanonicalizeMol(mol)

        return mol
    except Exception as e:
        return None


def to_simple_sdf(mol: Chem.Mol) -> str:
    """Convert an RDKit Mol to a simplified SDF string (atom + bond blocks only)."""
    mol_block = Chem.MolToMolBlock(mol, includeStereo=False)
    lines = mol_block.split("\n")
    simple_sdf = ""
    for line in lines:
        parts = line.split()
        if len(parts) == 4:  # bond line
            start_atom, end_atom, bond_order, stereoscopy = parts[0:4]
            simple_sdf += f"{start_atom} {end_atom} {bond_order} {stereoscopy}\n"
        elif len(parts) == 16:  # atom line
            x, y, z, atom = parts[0:4]
            charge = parts[5]
            simple_sdf += f"{x} {y} {z} {atom} {charge}\n"
    return simple_sdf


def to_indexed_sdf(mol: Chem.Mol) -> str:
    """Convert an RDKit Mol to an indexed (enumerated) simplified SDF string."""
    mol_block = Chem.MolToMolBlock(mol)
    lines = mol_block.split("\n")
    charges = sum([l.split()[3:] for l in lines if l.startswith("M  CHG")], [])
    charge_map = {int(c): int(charge) for c, charge in zip(charges[0::2], charges[1::2])}
    simple_sdf = ""
    atom_index = 1
    for line in lines:
        parts = line.split()
        if len(parts) == 16:  # atom line
            x, y, z, atom = parts[0:4]
            charge = charge_map.get(atom_index, 0) # default to 0 if not found
            simple_sdf += f"{atom_index} {x} {y} {z} {atom} {f"+{charge}" if charge > 0 else charge}\n"
            atom_index += 1
        if len(parts) == 4:  # bond line
            start_atom, end_atom, bond_order, stereoscopy = parts[0:4]
            simple_sdf += f"{start_atom} {end_atom} {bond_order} {stereoscopy}\n"
        
    return simple_sdf[:-1] # remove the last newline


def reconstruct_simple_sdf(simple_sdf: str) -> Optional[Chem.Mol]:
    """Reconstruct an RDKit Mol from a simplified ``<sdf>`` formatted string."""
    simple_sdf = simple_sdf.replace('<sdf>', '').replace('</sdf>', '')
    # Charges in the full SDF format are so unintuitive it's not even worth explaining,
    # so we just map them to normal charges.
    try:
        atom_lines = []
        bond_lines = []
        enumerated = False
        
        for line in simple_sdf.split("\n"):
            parts = line.split()
            if len(parts) == 5:  # atom line
                atom_lines.append(line)
            elif len(parts) == 6:  # enumerated atom line
                atom_lines.append(line)
                enumerated = True
            elif len(parts) == 0:  # skip empty lines
                continue
            else:  # bond line
                bond_lines.append(line)

        if enumerated:
            atom_lines = sorted(atom_lines, key=lambda x: int(x.split()[0]))

        full_sdf = "\n     AIGen          3D\n\n"
        full_sdf += f"{str(len(atom_lines)).rjust(3)}{str(len(bond_lines)).rjust(3)}  0     0  0  0  0  0  0999 V2000\n"
        charge_map = {}
        for i, line in enumerate(atom_lines):
            x, y, z, atom, charge = line.split()[-5:]
            charge_map[i+1] = str(int(charge))
            full_sdf += f"{x.rjust(10)}{y.rjust(10)}{z.rjust(10)} {atom.ljust(2)}  0  0  0  0  0  0  0  0  0  0  0  0\n"
        for line in bond_lines:
            start_atom, end_atom, bond_order, stereoscopy = line.split()[-4:]
            full_sdf += f"{start_atom.rjust(3)}{end_atom.rjust(3)}{bond_order.rjust(3)}{stereoscopy.rjust(3)}\n"
        charges = [c for c in charge_map.items() if c[1] != "0"]
        # Chunk charges into sublists of at most 8 elements each
        charge_lines = [charges[i:i+8] for i in range(0, len(charges), 8)]
        for charge_line in charge_lines:
            full_sdf += f"M  CHG  {len(charge_line)}{''.join([f"{str(i).rjust(4)}{charge.rjust(4)}" for i, charge in charge_line])}\n"
        full_sdf += "M  END\n"
        mol = Chem.MolFromMolBlock(full_sdf, sanitize=False)
        if mol and mol.GetNumAtoms() == 0:
            return None
        if mol and mol.GetNumConformers() == 0:
            return None
        return mol
    except Exception as e:
        return None


def reconstruct_full_sdf(full_sdf: str) -> Optional[Chem.Mol]:
    """Reconstruct an RDKit Mol from a full SDF/MolBlock string."""
    full_sdf = full_sdf.replace('<sdf>', '').replace('</sdf>', '')
    try:
        mol = Chem.MolFromMolBlock(full_sdf)
        if mol is None:
            return None
        if mol.GetNumAtoms() == 0:
            return None
        if mol.GetNumConformers() == 0:
            return None
        return mol
    except Exception as e:
        return None


def reconstruct_smiles_enum(s: str) -> Optional[Chem.Mol]:
    """Reconstruct an RDKit Mol from atom-mapped ``<smiles_xyz>`` formatted text."""
    s = s.replace('<smiles_xyz>', '').replace('</smiles_xyz>', '')
    try:
        lines = s.split("|")
        smiles, coords_lines = lines[0], lines[1:]
        mol = Chem.MolFromSmiles(smiles)
        mapnum_to_idx = {a.GetAtomMapNum(): a.GetIdx() for a in mol.GetAtoms()}
        conf = Chem.Conformer()
        coords_lines = [line for line in coords_lines if "H" not in line]
        for _, line in enumerate(coords_lines):
            map_num, _, x, y, z = line.split()
            conf.SetAtomPosition(mapnum_to_idx[int(map_num)], [float(x), float(y), float(z)])

        mol.AddConformer(conf)

        return mol
    except Exception as e:
        return None


def center_chem_obj(objs: list, rotate: bool = True, return_transform: bool = False, center_noise: float = 50.0):
    """Center (and optionally randomly rotate) a collection of RDKit Mols / ProDy objects."""
    coords = []
    for obj in objs:
        if obj is None:
            continue
        if isinstance(obj, Chem.Mol):
            obj_pos = obj.GetConformer().GetPositions()
            if obj_pos.shape[0]:
                coords.append(obj_pos)
        else:
            obj_pos = obj.getCoords()
            if obj_pos.shape[0]:
                coords.append(obj_pos)
    center = np.concatenate(coords).mean(axis=0)
    center = center + center_noise * np.random.randn(3)
    center = center

    rot_mx = np.eye(3)
    if rotate:
        rot_mx = special_ortho_group.rvs(3)
    rot_mx = rot_mx
        

    for obj in objs:
        if obj is None:
            continue
        if isinstance(obj, Chem.Mol):
            conf = obj.GetConformer()
            pos = conf.GetPositions()
            new_pos = (pos - center[None, :]) @ rot_mx
            for i in range(obj.GetNumAtoms()):
                x, y, z = new_pos[i]
                conf.SetAtomPosition(i, Point3D(x, y, z))
        else:
            pos = obj.getCoords()
            new_pos = (pos - center[None, :]) @ rot_mx
            obj.setCoords(new_pos)

    if return_transform:
        return (rot_mx, -center)

def valid_mol(mol: Optional[Chem.Mol]) -> bool:
    """Return True if *mol* is a non-empty, connected molecule."""
    if mol is None:
        return False
    if mol.GetNumAtoms() == 0:
        return False
    if mol.GetNumAtoms() < 100 and "." in Chem.MolToSmiles(mol):
        # Proteins can be disconnected, but not small molecules
        return False
    return True

def good_mol(mol: Chem.Mol) -> bool:
    """Return True if *mol* passes drug-like heuristics (size, element set)."""
    if mol.GetNumAtoms() < 25:
        return False
    if any((atom.GetAtomicNum() > 20) and (atom.GetAtomicNum() not in [35, 53]) for atom in mol.GetAtoms()):
        return False
    return True

def valid_protein(protein) -> bool:
    """Return True if *protein* (a ProDy AtomGroup) is non-empty."""
    if protein is None:
        return False
    if protein.numAtoms() == 0:
        return False
    return True