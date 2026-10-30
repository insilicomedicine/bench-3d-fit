"""Add prompts to a parquet dataset by converting the generation conditions to prompts.
Also stacks the datasets into a single parquet file for sharing on Hugging Face.

This script is not intended to be part of the usual pipeline of using the benchmark, this
was used to create the dataset for sharing on Hugging Face. The exact paper test sets are
versioned as ``data/test_sets/crossdocked.parquet`` and
``data/test_sets/plinder.parquet``.
"""

import argparse

import pandas as pd

from bench_3d_fit.generation_conditions import GenerationConditions

paper_cond_sets = {
    'pocket': ['pocket'],
    'pocket_fragment': ['pocket', 'anchor_points'],
    'pocket_pharmacophore': ['pocket', 'pharmacophore'],
    'pocket_fragment_pharmacophore_interaction': ['pocket', 'anchor_points', 'pharmacophore', 'pocket_ligand_interaction'],
}

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Add prompts to a parquet dataset by converting the generation conditions to prompts.",
            "Input is the path to the parquet file containing the generation conditions.",
            "Output is the path to the parquet file containing the prompts.",
        )
    )
    parser.add_argument(
        "--input-files",
        nargs="+",
        type=str,
        required=True,
        help="Paths to the input parquet files, one for each dataset. Must contain 'sdf', 'pdb', 'anchor_points', 'pharmacophore_points', and 'pli_points' columns.",
    )
    parser.add_argument(
        "--dataset-names",
        nargs="+",
        type=str,
        required=True,
        help="Names of the datasets, one for each input file.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Path to the output parquet file. Will contain same columns as input files and 'prompt' columns for each condition set used in the paper.",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    all_dfs = []
    for input_file, dataset_name in zip(args.input_files, args.dataset_names):
        df = pd.read_parquet(input_file)
        for index, row in df.iterrows():
            gen_conditions = GenerationConditions.from_test_row(row)
            for cond_set_name, cond_set_conditions in paper_cond_sets.items():
                prompt = gen_conditions.to_prompt(conditions_to_include=cond_set_conditions)
                df.at[index, f"prompt_{cond_set_name}"] = prompt
        df["source_dataset"] = dataset_name
        all_dfs.append(df)
    df = pd.concat(all_dfs, ignore_index=True)
    df.to_parquet(args.output_file)
if __name__ == "__main__":
    main()