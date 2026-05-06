"""Formatting helpers that convert 3D structural data into LLM-readable prompt text."""

import pandas as pd
from typing import Any, Dict, List


def format_environment_pdb(environment_pdb: str) -> str:
    """Format a PDB-text protein pocket into a human-readable prompt block."""
    atom_data = {
        "atom_symbol": [],
        "res_name": [],
        "chain_id": [],
        "res_id": [],
        "x": [],
        "y": [],
        "z": []
    }
    for line in environment_pdb.split("\n"):
        if line.startswith("ATOM"):
            atom_symbol = line[12:16].strip()
            if not atom_symbol.startswith("H"):
                atom_data["atom_symbol"].append(line[12:16].strip())
                atom_data["res_name"].append(line[17:20].strip())
                atom_data["chain_id"].append(line[21:22].strip())
                atom_data["res_id"].append(line[22:26].strip())
                atom_data["x"].append(line[30:38].strip())
                atom_data["y"].append(line[38:46].strip())
                atom_data["z"].append(line[46:54].strip())
    atom_data = pd.DataFrame(atom_data)
    pdb_str = "Atoms are grouped first by chain, then by residue, and are referenced using atom names from PDB-style protein notation.\n"
    pdb_str += "Format: <pdb_atom_name> at (<x>, <y>, <z>)\n"
    for _, chain_group in atom_data.groupby(["chain_id"], sort=False):
        pdb_str += f"Chain {chain_group.iloc[0]['chain_id']}:\n"
        for _, res_group in chain_group.groupby(["res_id"], sort=False):
            pdb_str += f"Residue {res_group.iloc[0]['res_id']}, {res_group.iloc[0]['res_name']}:\n"
            for _, row in res_group.iterrows():
                pdb_str += f"{row['atom_symbol']} at ({row['x']}, {row['y']}, {row['z']})\n"
    return pdb_str[:-1]  # remove the last newline

def format_pharmacophore_points(pharmacophore_points: List[Dict[str, Any]]) -> str:
    """Format pharmacophore points into a human-readable prompt block."""
    code_to_name = {
        "D": "HBDonor",
        "A": "HBAcceptor",
        "H": "Hydrophobic",
        "P": "PositiveIonizable",
        "N": "NegativeIonizable",
        "a": "Aromatic",
        "e": "Exclusion"
    }
    name_to_code = {
        "HBDonor": "D",
        "HBAcceptor": "A",
        "Hydrophobic": "H",
        "PositiveIonizable": "P",
        "NegativeIonizable": "N",
        "Aromatic": "a",
        "Exclusion": "e"
    }
    pharmacophore_type_descriptions = {
        "HBDonor": "Hydrogen bond donor",
        "HBAcceptor": "Hydrogen bond acceptor",
        "Hydrophobic": "Hydrophobic group",
        "PositiveIonizable": "Positive ionizable group",
        "NegativeIonizable": "Negative ionizable group",
        "Aromatic": "Aromatic group",
        "Exclusion": "Exclusion group"
    }
    return "Format: <pharmacophore_type> within 0.5 Angstroms of (<x>, <y>, <z>)\n" + "\n".join(
        [f"{pharmacophore_type_descriptions[px['name']]} within {px['radius']} Angstroms of ({px['x']:.3f}, {px['y']:.3f}, {px['z']:.3f})" \
            for px in pharmacophore_points]
    )

def format_anchor_points(anchor_points: List[Dict[str, Any]]) -> str:
    """Format anchor points into a human-readable prompt block."""
    periodic_table = {
        "H": "Hydrogen (H)",
        "C": "Carbon (C)",
        "N": "Nitrogen (N)",
        "O": "Oxygen (O)",
        "F": "Fluorine (F)",
        "S": "Sulfur (S)",
        "P": "Phosphorus (P)",
        "Cl": "Chlorine (Cl)",
        "Br": "Bromine (Br)",
        "I": "Iodine (I)",
        "Si": "Silicon (Si)",
        "As": "Arsenic (As)",
        "Se": "Selenium (Se)",
    }
    return "Format: Element Name (<atomic_symbol>) within 0.5 Angstroms of (<x>, <y>, <z>)\n" + "\n".join(
        [f"{periodic_table.get(ap['name'], ap['name'])} within 0.5 Angstroms of ({ap['x']:.3f}, {ap['y']:.3f}, {ap['z']:.3f})" \
            for ap in anchor_points]
    )

def format_pocket_ligand_interaction_points(pocket_ligand_interaction_points: List[Dict[str, str]]) -> str:
    """Format pocket-ligand interaction points into a human-readable prompt block."""
    pli_type_descriptions = {
        "Hydrophobic": "Hydrophobic",
        "HBDonor": "Hydrogen bond",
        "HBAcceptor": "Hydrogen bond",
        "PiStacking": "Pi-Stacking",
        "Anionic": "Anionic",
        "Cationic": "Cationic",
        "CationPi": "Cation-pi",
        "PiCation": "Pi-cation",
        "VdWContact": "Van der Waals",
    }
    descriptions = []
    res_ids = []
    chains = []
    res_names = []
    for point in pocket_ligand_interaction_points:
        descriptions.append(pli_type_descriptions[point['name']])
        res_ids.append(point['residue'].split('.')[0][3:])
        chains.append(point['residue'].split('.')[1])
        res_names.append(point['residue'].split('.')[0][:3])
    return "\n".join(
        [f"{description} interaction with any atom of residue {res_id}, {res_name} from chain {chain}" \
            for description, res_id, chain, res_name in zip(descriptions,
                                                                   res_ids, chains,
                                                                   res_names)]
    )

def format_output_instructions(format_instructions: str) -> str:
    """Build the output-format instruction block for the given format type.

    Supported formats: ``'smiles_xyz'``, ``'sdf'``, ``'sdf_enum'``,
    ``'smiles_enum'``, ``'smiles_edits'``.
    """
    prompt = ""
    if format_instructions == "smiles_xyz":
        prompt += "In this format, the molecule is first provided as a SMILES string, then each heavy atom (non-hydrogen atom) is listed with its 3D coordinates in Angstroms with 3 decimal places.\n"
        prompt += "The number of heavy atoms in the SMILES and following coordinates should be the same, all atoms should be described in the same order as in SMILES.\n"
        prompt += "For example, a molecule with the SMILES string 'Cc1cc(CC2CCC(F)(F)C2)n(Cc2cscn2)c1' with (<x>, <y>, <z>) coordinates for each atom can be output as:\n"
        prompt += "<smiles_xyz>Cc1cc(CC2CCC(F)(F)C2)n(Cc2cscn2)c1|C 30.987 6.655 24.653|C 31.351 7.912 24.029|C 30.624 8.775 23.180|C 31.510 9.726 22.702|C 31.368 10.813 21.681|C 30.007 11.245 21.155|C 28.976 11.750 22.162|C 27.672 11.818 21.343|C 27.928 11.007 20.084|F 26.905 10.209 19.755|F 28.130 11.857 19.044|C 29.209 10.235 20.333|N 32.716 9.446 23.320|C 34.031 10.014 23.112|C 35.271 9.181 23.283|C 35.399 7.840 22.977|S 36.876 7.257 23.558|C 37.373 8.844 23.863|N 36.438 9.775 23.782|C 32.587 8.443 24.204</smiles_xyz>\n"
    elif format_instructions == "sdf":
        prompt += "In this format, the molecule is provided as a simplified SDF/MolBlock string. Which has two parts, an atom block and a bond block.\n"
        prompt += "The atom block lists each heavy atom (non-hydrogen atom), in any order, starting with its x y z coordinates in Angstroms with 3 decimal places, followed by the atomic symbol and the charge on the atom.\n"
        prompt += "The bond block lists each bond, containing the start and end atoms, by index from the atom block (indexed from 1), the bond order (1 for single, 2 for double, 3 for triple) and finally the stereoscopy of the bond (0 for none, 1 for up, 6 for down)\n"
        prompt += "For example, a molecule with the SMILES string 'Cc1cc(CC2CCC(F)(F)C2)n(Cc2cscn2)c1' with (<x>, <y>, <z>) coordinates for each atom can be output as:\n"
        prompt += "<sdf>30.987 6.655 24.653 C 0\n31.351 7.912 24.029 C 0\n30.624 8.775 23.180 C 0\n31.510 9.726 22.702 C 0\n31.368 10.813 21.681 C 0\n30.007 11.245 21.155 C 0\n28.976 11.750 22.162 C 0\n27.672 11.818 21.343 C 0\n27.928 11.007 20.084 C 0\n26.905 10.209 19.755 F 0\n28.130 11.857 19.044 F 0\n29.209 10.235 20.333 C 0\n32.716 9.446 23.320 N 0\n34.031 10.014 23.112 C 0\n35.271 9.181 23.283 C 0\n35.399 7.840 22.977 C 0\n36.876 7.257 23.558 S 0\n37.373 8.844 23.863 C 0\n36.438 9.775 23.782 N 0\n32.587 8.443 24.204 C 0\n1 2 1 0\n2 3 1 0\n3 4 2 0\n4 5 1 0\n5 6 1 0\n6 7 1 0\n7 8 1 0\n8 9 1 0\n9 10 1 0\n9 11 1 0\n9 12 1 0\n4 13 1 0\n13 14 1 0\n14 15 1 0\n15 16 2 0\n16 17 1 0\n17 18 1 0\n18 19 2 0\n13 20 1 0\n20 2 2 0\n12 6 1 0\n19 15 1 0</sdf>\n"
    elif format_instructions == "sdf_enum":
        prompt += "In this format, the molecule is provided as a simplified SDF/MolBlock string. Which has two parts, an atom block and a bond block.\n"
        prompt += "The atom block contains a line for each heavy atom (non-hydrogen atom), 6 values per line. First is the atom index, starting from 1, followed 3 values for the x y z coordinates in Angstroms with 3 decimal places each, fifth is the atomic symbol and sixth is the charge on the atom (0 for no charge, -1 for negative charge, +1 for positive charge)\n"
        prompt += "The bond block contains a line for each bond, 4 values per line. First is the start atom index from the atom block, the second is the end atom index from the atom block, the third is the bond order (1 for single, 2 for double, 3 for triple) and fourth is the stereoscopy of the bond (if applicable, 0 for none, 1 for up, 6 for down).\n"
        prompt += "Bonds do not have a unique identifier index, the start and end atom indices are used to identify the bond. Hydrogen atoms are not included in this format and are instead determined implicity from the avaliable valencies of the listed heavy atoms \n"
        prompt += "For example, a molecule with the SMILES string 'Cc1cc(CC2CCC(F)(F)C2)n(Cc2cscn2)c1' with (<x>, <y>, <z>) coordinates for each atom can be output as:\n"
        prompt += "<sdf>1 30.9870 6.6550 24.6530 C 0\n2 31.3510 7.9120 24.0290 C 0\n3 30.6240 8.7750 23.1800 C 0\n4 31.5100 9.7260 22.7020 C 0\n5 31.3680 10.8130 21.6810 C 0\n6 30.0070 11.2450 21.1550 C 0\n7 28.9760 11.7500 22.1620 C 0\n8 27.6720 11.8180 21.3430 C 0\n9 27.9280 11.0070 20.0840 C 0\n10 26.9050 10.2090 19.7550 F 0\n11 28.1300 11.8570 19.0440 F 0\n12 29.2090 10.2350 20.3330 C 0\n13 32.7160 9.4460 23.3200 N 0\n14 34.0310 10.0140 23.1120 C 0\n15 35.2710 9.1810 23.2830 C 0\n16 35.3990 7.8400 22.9770 C 0\n17 36.8760 7.2570 23.5580 S 0\n18 37.3730 8.8440 23.8630 C 0\n19 36.4380 9.7750 23.7820 N 0\n20 32.5870 8.4430 24.2040 C 0\n1 2 1 0\n2 3 1 0\n3 4 2 0\n4 5 1 0\n5 6 1 0\n6 7 1 0\n7 8 1 0\n8 9 1 0\n9 10 1 0\n9 11 1 0\n9 12 1 0\n4 13 1 0\n13 14 1 0\n14 15 1 0\n15 16 2 0\n16 17 1 0\n17 18 1 0\n18 19 2 0\n13 20 1 0\n20 2 2 0\n12 6 1 0\n19 15 1 0</sdf>\n"
    elif format_instructions == "smiles_enum":
        prompt += "For molecule with N heavy (non-hydrogen) atoms the format is <smiles_xyz>Atom-Mapped SMILES|idx1 Atom1 x y z|...|idxN AtomN x y z</smiles_xyz>. The coordinate with idx i entry is corresponded to the i-th heavy atom in the atom mapping. The order of atoms can be arbitrary. The SMILES must represent a single connected valid molecule. Provide coordinates for every heavy atom only, and the number of coordinate entries must exactly match the number of heavy atoms in the SMILES, the atom element for entry with idx i should match the i-th heavy atom in SMILES. The answer shouldn't contain any additional, omitted or duplicated atoms. The bond lengths and angles must follow chemical rules such that the strain energy of the molecule is minimal. \n"
        prompt += "Important: Before finalizing, explicitly verify: SMILES is a single connected molecule; count the number of heavy atoms in SMILES and coordinate section and check that heavy-atom count in SMILES equals number of coordinate records; enumerate heavy atoms in the SMILES and check that coordinate records cover all SMILES atoms, every atom symbol for entry with index i in coordinates matches the corresponding i-th SMILES heavy atom; check that valence of every atom is valid, no atom is omitted or duplicated. If uncertain, prefer a molecular ligand output validity over satisfying other constraints. If you can not satisfy the same order as in SMILES, give the arbitrary order of coordinates, just check the consistency of indexes of atoms with SMILES. Do not output any molecule unless all checks pass. If any check fails, regenerate silently and output only the corrected final molecule. Do not output any molecule unless all checks pass. \n"
        prompt += "For example, a 3D ligand molecule with the SMILES string 'Cc1cc(CC2CCC(F)(F)C2)n(Cc2cscn2)c1' with (<x>, <y>, <z>) coordinates for each atom can be output as:\n"
        prompt += "<smiles_xyz>[C:1][c:2]1[c:3][c:4]([C:5][C:6]2[C:7][C:8][C:9]([F:10])([F:11])[C:12]2)[n:13]([C:14][c:15]2[c:16][s:17][c:18][n:19]2)[c:20]1|1 C 30.987 6.655 24.653|2 C 31.351 7.912 24.029|3 C 30.624 8.775 23.180|4 C 31.510 9.726 22.702|5 C 31.368 10.813 21.681|6 C 30.007 11.245 21.155|7 C 28.976 11.750 22.162|8 C 27.672 11.818 21.343|9 C 27.928 11.007 20.084|10 F 26.905 10.209 19.755|11 F 28.130 11.857 19.044|12 C 29.209 10.235 20.333|13 N 32.716 9.446 23.320|14 C 34.031 10.014 23.112|15 C 35.271 9.181 23.283|16 C 35.399 7.840 22.977|17 S 36.876 7.257 23.558|18 C 37.373 8.844 23.863|19 N 36.438 9.775 23.782|20 C 32.587 8.443 24.204</smiles_xyz>\n"
        prompt += "or the same molecule with the same conformation but with the atom coordinates in a different order can be output as:\n"
        prompt += "<smiles_xyz>[C:1][c:2]1[c:3][c:4]([C:5][C:6]2[C:7][C:8][C:9]([F:10])([F:11])[C:12]2)[n:13]([C:14][c:15]2[c:16][s:17][c:18][n:19]2)[c:20]1|14 C 34.031 10.014 23.112|11 F 28.130 11.857 19.044|13 N 32.716 9.446 23.320|3 C 30.624 8.775 23.180|20 C 32.587 8.443 24.204|1 C 30.987 6.655 24.653|4 C 31.510 9.726 22.702|15 C 35.271 9.181 23.283|8 C 27.672 11.818 21.343|6 C 30.007 11.245 21.155|12 C 29.209 10.235 20.333|2 C 31.351 7.912 24.029|17 S 36.876 7.257 23.558|5 C 31.368 10.813 21.681|19 N 36.438 9.775 23.782|18 C 37.373 8.844 23.863|7 C 28.976 11.750 22.162|10 F 26.905 10.209 19.755|9 C 27.928 11.007 20.084|16 C 35.399 7.840 22.977</smiles_xyz>\n"
    elif format_instructions == "smiles_edits":
        prompt += "For 3D molecule the format is <smiles_xyz>Atom-Mapped SMILES|i1 (i1-th Heavy Atom Symbol) (i1-th Heavy Atom Coordinates)|...|iK (iK-th Heavy Atom Symbol) (iK-th Heavy Atom Coordinates)</smiles_xyz>. The order of atoms can be arbitrary, just ensure that all heavy atoms in SMILES has coordinates. Moreover, you can introduce several coordinates entries for the same atom, only final one will be considered, the first appearances will be ignored. You can use this to correct the atom positions by overwriting their coordinates. The SMILES must represent a single connected valid molecule. Provide coordinates for every heavy atom only, the atom symbol for entry with idx iK should match the iK-th heavy atom in SMILES. The answer shouldn't contain any additional atoms except ones presented in SMILES, all heavy atoms in SMILES should have at least one entry with coordinates. The bond lengths and angles must follow chemical rules such that the strain energy of the molecule is minimal. \n"
        prompt += "For example, a 3D ligand molecule with the SMILES string 'Cc1cc(CC2CCC(F)(F)C2)n(Cc2cscn2)c1' with (<x>, <y>, <z>) coordinates for each atom can be output as:\n"
        prompt += "<smiles_xyz>[C:1][c:2]1[c:3][c:4]([C:5][C:6]2[C:7][C:8][C:9]([F:10])([F:11])[C:12]2)[n:13]([C:14][c:15]2[c:16][s:17][c:18][n:19]2)[c:20]1|1 C 30.987 6.655 24.653|2 C 31.351 7.912 24.029|3 C 30.624 8.775 23.180|4 C 31.510 9.726 22.702|5 C 31.368 10.813 21.681|6 C 30.007 11.245 21.155|7 C 28.976 11.750 22.162|8 C 27.672 11.818 21.343|9 C 27.928 11.007 20.084|10 F 26.905 10.209 19.755|11 F 28.130 11.857 19.044|12 C 29.209 10.235 20.333|13 N 32.716 9.446 23.320|14 C 34.031 10.014 23.112|15 C 35.271 9.181 23.283|16 C 35.399 7.840 22.977|17 S 36.876 7.257 23.558|18 C 37.373 8.844 23.863|19 N 36.438 9.775 23.782|20 C 32.587 8.443 24.204</smiles_xyz>\n"
        prompt += "or the same molecule with draft coordinates entries for 5-th and 10-th atoms\n"
        prompt += "<smiles_xyz>[C:1][c:2]1[c:3][c:4]([C:5][C:6]2[C:7][C:8][C:9]([F:10])([F:11])[C:12]2)[n:13]([C:14][c:15]2[c:16][s:17][c:18][n:19]2)[c:20]1|1 C 30.987 6.655 24.653|2 C 31.351 7.912 24.029|3 C 30.624 8.775 23.180|4 C 31.510 9.726 22.702|5 C 31.123 10.032 22.001|6 C 30.007 11.245 21.155|7 C 28.976 11.750 22.162|8 C 27.672 11.818 21.343|9 C 27.928 11.007 20.084|10 F 26.398 9.909 20.105|11 F 28.130 11.857 19.044|12 C 29.209 10.235 20.333|13 N 32.716 9.446 23.320|14 C 34.031 10.014 23.112|15 C 35.271 9.181 23.283|16 C 35.399 7.840 22.977|17 S 36.876 7.257 23.558|18 C 37.373 8.844 23.863|19 N 36.438 9.775 23.782|20 C 32.587 8.443 24.204|5 C 31.368 10.813 21.681|10 F 26.905 10.209 19.755</smiles_xyz>\n"
        prompt += "or the same molecule with the same conformation but with the atom coordinates in a different order can be output as:\n"
        prompt += "<smiles_xyz>[C:1][c:2]1[c:3][c:4]([C:5][C:6]2[C:7][C:8][C:9]([F:10])([F:11])[C:12]2)[n:13]([C:14][c:15]2[c:16][s:17][c:18][n:19]2)[c:20]1|14 C 34.031 10.014 23.112|11 F 28.130 11.857 19.044|13 N 32.716 9.446 23.320|3 C 30.624 8.775 23.180|20 C 32.587 8.443 24.204|1 C 30.987 6.655 24.653|4 C 31.510 9.726 22.702|15 C 35.271 9.181 23.283|8 C 27.672 11.818 21.343|6 C 30.007 11.245 21.155|12 C 29.209 10.235 20.333|2 C 31.351 7.912 24.029|17 S 36.876 7.257 23.558|5 C 31.368 10.813 21.681|19 N 36.438 9.775 23.782|18 C 37.373 8.844 23.863|7 C 28.976 11.750 22.162|10 F 26.905 10.209 19.755|9 C 27.928 11.007 20.084|16 C 35.399 7.840 22.977</smiles_xyz>\n"
    else:
        raise ValueError(f"Invalid format instructions: {format_instructions}. The valid format instructions are: smiles_xyz, sdf, sdf_enum, smiles_enum, smiles_edits")
    return prompt