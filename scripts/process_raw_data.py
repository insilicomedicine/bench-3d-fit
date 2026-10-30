"""
---------------------------------------------------------------------------------------
                Functions for Processing Raw Data From Source
Applied to CrossDocked2020 and PLInder datasets, but kept as general as possible
to be applied to other datasets as desired.
Processed output defaults to ``data/test_sets/{dataset}.parquet``.
---------------------------------------------------------------------------------------
"""
import argparse
import gzip
from io import StringIO
import os
from pathlib import Path

import pandas as pd
import prody
from rdkit import Chem

from bench_3d_fit.mol_io import (
    center_chem_obj,
    good_mol,
    valid_mol,
    valid_protein,
)

"""
---------------------------------------------------------------------------------------
                Functions for Processing CrossDocked Raw Data
Crossdocked 2020 project: https://github.com/gnina/models/blob/master/data/CrossDocked2020/README.md
Downloaded from: https://bits.csb.pitt.edu/files/crossdock2020/ on 2026-04-20
We specifically use the DOWNSAMPLED version provided:
https://bits.csb.pitt.edu/files/crossdock2020/downsampled_CrossDocked2020_v1.3.tgz
This contains a directory of split files:
https://bits.csb.pitt.edu/files/crossdock2020/downsampled_CrossDocked2020_v1.3_types.tgz
We are using test split 0 from this directory, named it2_tt_v1.3_10p20n_test0.types.
("Split File" in CLI usage)

So, to run this script, you need to:
1. Download the CrossDocked 2020 dataset from the provided link.
2. Unzip the dataset to the directory of your choice.
3. Run the script with the following command:
python process_raw_data.py --dataset crossdocked
    --input-dir /path/to/input/directory --split-file /path/to/split/file
---------------------------------------------------------------------------------------
"""

def get_crossdocked_test_df(split_file: str):
    split_df = pd.read_csv(
        split_file,
        sep=' ',
        header=None,
        names=["label", "pK", "RMSD", "Receptor", "Ligand", "Vina Score"]
    )
    split_df["Vina Score"] = split_df["Vina Score"].apply(lambda x: float(x[1:]))
    split_df = split_df[split_df['Vina Score'] < -6]  # Ensure decent protein-ligand binding (more likely to have relevant pharmacophore/mandatory point features)
    split_df = split_df[split_df['RMSD'] < 0.5]  # Only taking complexes that are VERY close to the crystal structure

    return split_df


def process_crossdocked_row(row: pd.Series, base_dir: str):
    system_id = row['Receptor']
    receptor_file = os.path.join(base_dir, system_id)
    receptor_file = "_".join(receptor_file.split(".")[0].split("_")[:-1])+".pdb"
    ligand_file = os.path.join(base_dir, row['Ligand'])
    ligand_file = "_".join(ligand_file.split(".")[0].split("_")[:-1])+".sdf.gz"
    if not os.path.exists(receptor_file):
        return None, None
    if not os.path.exists(ligand_file):
        return None, None

    # -------------- Test loading the ligand and receptor --------------
    with gzip.open(ligand_file, 'r') as f:
        sdf_contents = f.read().decode('utf-8')
    with open(receptor_file, 'r') as f:
        pdb_contents = f.read()
    # Tag the molecule as 3D to avoid RDKit warnings
    sdf_contents = "\n".join(
        sdf_contents.split("\n")[:1] + ["3D"] + sdf_contents.split("\n")[2:]
    )
    return sdf_contents, pdb_contents


"""
---------------------------------------------------------------------------------------
                Functions for Processing PLInder Raw Data
PLInder project: https://www.plinder.sh/
Downloaded from https://console.cloud.google.com/storage/browser/plinder on 2026-04-20
To get the splits, we download the following file ("Split File" in CLI usage):
plinder/2024-06/v2/splits/split.parquet
As well as all of the systems in the directory ("Input Directory" in CLI usage):
plinder/2024-06/v2/systems/
(all of the systems are stored as about 1000 different zip files, they are 
unzipped into the systems_dir)

So, to run this script, you need to:
1. Download the PLInder dataset from the provided link.
2. Unzip all of the zipped system files into the input directory.
3. Run the script with the following command:
python process_raw_data.py --dataset plinder
    --input-dir /path/to/input/directory --split-file /path/to/split/file
---------------------------------------------------------------------------------------
"""

def get_plinder_test_df(split_file: str):
    split_df = pd.read_parquet(split_file)
    test_df = split_df[split_df['split'] == 'test']
    return test_df

def process_plinder_row(row: pd.Series, base_dir: str):
    system_id = row['system_id']
    system_file = os.path.join(base_dir, system_id)
    if not os.path.exists(system_file):
        return None, None
    if not os.path.exists(os.path.join(system_file, 'receptor.pdb')):
        return None, None
    if not os.path.exists(os.path.join(system_file, 'ligand_files')):
        return None, None
    ligand_dir = os.path.join(system_file, 'ligand_files')
    # Find all SDF files and their sizes

    sdf_files = [
        f for f in os.listdir(ligand_dir)
        if f.endswith('.sdf') and os.path.isfile(os.path.join(ligand_dir, f))
    ]
    if not sdf_files:
        return None, None
    # Find the largest file by size - Ignoring cofactors, Metal Ions, etc.
    largest_sdf_file = max(
        sdf_files,
        key=lambda f: os.path.getsize(os.path.join(ligand_dir, f))
    )
    largest_sdf_path = os.path.join(ligand_dir, largest_sdf_file)
    with open(largest_sdf_path, 'r') as f:
        sdf_contents = f.read()

    with open(os.path.join(system_file, 'receptor.pdb'), 'r') as f:
        pdb_contents = f.read()

    return sdf_contents, pdb_contents

"""
---------------------------------------------------------------------------------------
                Main Script
Shared Functionality between CrossDocked and PLInder datasets.
---------------------------------------------------------------------------------------
"""

DEFAULT_TEST_DATA_ROOT = (
    Path(__file__).resolve().parents[1] / "data" / "test_sets"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Process raw protein-ligand complex data from CrossDocked2020 or PLInder."
            "Input is the root directory for each dataset, as downloaded from the source."
            "Output is a parquet file containing the processed data."
        )
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["crossdocked", "plinder"],
        help="Which dataset to process.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_TEST_DATA_ROOT,
        help=(
            "Directory where {dataset}.parquet will be written. "
            "Defaults to the repository's data/test_sets directory."
        ),
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Root directory containing the raw dataset files.",
    )
    parser.add_argument(
        "--split-file",
        type=str,
        required=True,
        help="Path to the split/types file that defines the test set.",
    )
    parser.add_argument(
        "--pocket-distance",
        type=float,
        default=10.0,
        help="Distance in angstroms from the center of the ligand to the protein to be considered a pocket.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.dataset == 'plinder':
        df_function = get_plinder_test_df
        row_function = process_plinder_row
    elif args.dataset == 'crossdocked':
        df_function = get_crossdocked_test_df
        row_function = process_crossdocked_row
    else:
        raise ValueError(f"Invalid dataset name:  args.dataset")

    test_df = df_function(args.split_file)
    output_dict = {
        'sdf': [],
        'pdb': [],
    }
    rejected_count = 0
    for i in range(len(test_df)):
        row = test_df.iloc[i]
        sdf_str, pdb_str = row_function(row, args.input_dir)
        mol = Chem.MolFromMolBlock(sdf_str, removeHs=False)
        if not valid_mol(mol):
            rejected_count += 1
            continue
        if not good_mol(mol):
            continue
        receptor = prody.parsePDBStream(StringIO(pdb_str))
        if not valid_protein(receptor):
            rejected_count += 1
            continue
        sel = receptor.select(
            f'same residue as ((protein or ion) and within {args.pocket_distance} of center)',
            center=mol.GetConformer().GetPositions().mean(axis=0)
        )
        if not valid_protein(sel):
            rejected_count += 1
            continue
        pdb_text = StringIO()
        prody.writePDBStream(pdb_text, sel)
        protein_mol = Chem.MolFromPDBBlock(pdb_text.getvalue(), removeHs=False)
        if not valid_mol(protein_mol):
            rejected_count += 1
            continue
        center_chem_obj(
            [protein_mol, mol], rotate=False, center_noise=50.0, return_transform=False
        )
        final_sdf = Chem.MolToMolBlock(mol)
        final_pdb = Chem.MolToPDBBlock(protein_mol)
        if final_sdf is None or final_pdb is None:
            print("Final molecules are None")
            rejected_count += 1
            continue
        mol_check = Chem.MolFromMolBlock(final_sdf)
        if not valid_mol(mol_check):
            rejected_count += 1
            continue
        protein_check_mol = Chem.MolFromPDBBlock(final_pdb, removeHs=False)
        if not valid_mol(protein_check_mol):
            rejected_count += 1
            continue
        protein_check_prody = prody.parsePDBStream(StringIO(final_pdb))
        if not valid_protein(protein_check_prody):
            rejected_count += 1
            continue
        output_dict['sdf'].append(final_sdf)
        output_dict['pdb'].append(final_pdb)

    df = pd.DataFrame(output_dict)
    print(f"Rejected count: {rejected_count}")
    print(f"Total complexes: {len(df)}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_dir / f"{args.dataset}.parquet")

if __name__ == "__main__":
    main()