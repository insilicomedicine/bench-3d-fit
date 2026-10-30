"""Pre-sample generation conditions for every row in a parquet test set.

Reads a parquet file with ``sdf`` and ``pdb`` columns, samples anchor points,
pharmacophore points, and pocket-ligand interaction points for each row, and
writes the augmented parquet with the new JSON columns. Input and output
default to ``data/test_sets/{dataset}.parquet``.
"""

import argparse
from pathlib import Path

import pandas as pd
import json
from bench_3d_fit.sampling import sample_generation_conditions

DEFAULT_TEST_DATA_ROOT = (
    Path(__file__).resolve().parents[1] / "data" / "test_sets"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Sample generation conditions from a parquet file and save them to a new (or overwrite existing) parquet file.",
        )
    )
    parser.add_argument(
        "--dataset",
        choices=["crossdocked", "plinder"],
        default="crossdocked",
        help="Bundled paper test set to process. Defaults to crossdocked.",
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=None,
        help=(
            "Input parquet containing 'sdf' and 'pdb'. Defaults to "
            "data/test_sets/{dataset}.parquet."
        ),
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help=(
            "Output parquet with sampled conditions. Defaults to the selected "
            "data/test_sets/{dataset}.parquet; use --overwrite for in-place "
            "updates."
        ),
    )
    parser.add_argument(
        "--min-ph4-features",
        type=int,
        default=1,
        help="Minimum number of pharmacophore features to sample.",
    )
    parser.add_argument(
        "--max-ph4-features",
        type=int,
        default=1,
        help="Maximum number of pharmacophore features to sample.",
    )
    parser.add_argument(
        "--pharmacophore-radius",
        type=float,
        default=0.5,
        help="Radius of the pharmacophore points.",
    )
    parser.add_argument(
        "--preferred-min-frag-size",
        type=int,
        default=3,
        help="Minimum size of the fragment to sample. Will be ignored if no fragment can be found.",
    )
    parser.add_argument(
        "--preferred-max-frag-size",
        type=int,
        default=8,
        help="Maximum size of the fragment to sample. Will be ignored if no fragment can be found.",
    )
    parser.add_argument(
        "--anchor-radius",
        type=float,
        default=0.5,
        help="Radius of the anchor points.",
    )
    parser.add_argument(
        "--min-pli-points",
        type=int,
        default=1,
        help="Minimum number of PLI points to sample.",
    )
    parser.add_argument(
        "--max-pli-points",
        type=int,
        default=1,
        help="Maximum number of PLI points to sample.",
    )
    parser.add_argument(
        "--config-sampling-seed",
        type=int,
        default=42,
        help="Seed for the random number generator.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file if it exists.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    default_path = DEFAULT_TEST_DATA_ROOT / f"{args.dataset}.parquet"
    input_file = default_path if args.input_file is None else args.input_file
    output_file = default_path if args.output_file is None else args.output_file
    df = pd.read_parquet(input_file)
    if output_file.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output file {output_file} already exists. "
            "Use --overwrite to overwrite."
        )
    anchor_points = []
    pharmacophore_points = []
    pli_points = []
    for _, row in df.iterrows():
        gen_conditions = sample_generation_conditions(
            row,
            min_ph4_features=args.min_ph4_features,
            max_ph4_features=args.max_ph4_features,
            pharmacophore_radius=args.pharmacophore_radius,
            preferred_min_frag_size=args.preferred_min_frag_size,
            preferred_max_frag_size=args.preferred_max_frag_size,
            anchor_radius=args.anchor_radius,
            min_pli_points=args.min_pli_points,
            max_pli_points=args.max_pli_points,
            config_sampling_seed=args.config_sampling_seed
        )
        anchor_points.append(json.dumps(gen_conditions.anchor_points))
        pharmacophore_points.append(json.dumps(gen_conditions.pharmacophore_points))
        pli_points.append(json.dumps(gen_conditions.pocket_ligand_interaction_points))
    df["anchor_points"] = anchor_points
    df["pharmacophore_points"] = pharmacophore_points
    df["pli_points"] = pli_points
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_file)
    
if __name__ == "__main__":
    main()