"""Example: score generated molecules using the 3D-Fit Benchmark library.

Demonstrates the typical workflow:
1. Load a test-set parquet.
2. Retrieve generation conditions for each row.
3. Generate molecules (placeholder: copies of the seed molecule).
4. Score the molecules with :class:`~bench_3d_fit.reward.scorer.RewardScorer`.
5. Write scored results and aggregated success statistics.
"""
from pathlib import Path
import argparse
import pandas as pd
import copy
from rdkit import Chem

from bench_3d_fit import (
    RewardScorer,
    GenerationConditions,
    get_generation_conditions,
    write_generation_results,
    write_success_stats,
)

DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
DEFAULT_TEST_DATA_ROOT = DEFAULT_DATA_ROOT / "test_sets"
DEFAULT_REFERENCE_ROOT = DEFAULT_DATA_ROOT / "novelty_reference"


def generate_molecules(generation_conditions: GenerationConditions) -> list[Chem.Mol]:
    """
    Call your generative model (LLM, diffusion, external binary, ...).

    Use any/all of the information in the generation conditions to generate molecules.
    Conditions provided in the generation conditions are used to score the molecules.

    Examples:
    - For an LLM, can do something like:

    generations = your_model(generation_conditions.to_prompt())
    
    and then pass the completions to the RewardScorer, which can parse the completions
    if the format instructions were followed by the model and score the molecules.

    - For a diffusion model that supports pocket and anchor point conditioning,
    can do something like:

    environment_pdb = generation_conditions.environment_pdb
    anchor_points = generation_conditions.anchor_points

    generations = your_model(environment_pdb, anchor_points)

    and then pass the reconstructed molecules to the RewardScorer, which will score the
    molecules.
    """
    return [copy.deepcopy(generation_conditions.seed_molecule) for _ in range(2)]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Example usage of the 3D-Fit Benchmark library to score generated "
            "molecules. The placeholder `generate_molecules` function generates "
            "two copies of the seed molecule from the generation conditions."
        )
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=None,
        help=(
            "Path to the test-set parquet. Defaults to "
            "data/test_sets/{dataset}.parquet."
        ),
    )
    parser.add_argument(
        "--results-path",
        type=str,
        required=True,
        help="Path to the directory where the scored molecules will be written.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["crossdocked", "plinder"],
        default="crossdocked",
        help=(
            "Dataset name. Defaults to crossdocked. Together with --condition, "
            "writes joint success, diversity, uniqueness, and novelty."
        ),
    )
    parser.add_argument(
        "--condition",
        type=str,
        default="all_conditions",
        help=(
            "Condition token (all_conditions, pocket, pocket_fragment, "
            "pocket_pharmacophore, or pocket_interaction). Defaults to "
            "all_conditions and writes the set-level aggregate scores."
        ),
    )
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=DEFAULT_REFERENCE_ROOT,
        help=(
            "Directory containing crossdocked_unique.txt and "
            "plinder_unique.txt for novelty scoring. Defaults to "
            "data/novelty_reference/."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_path = (
        DEFAULT_TEST_DATA_ROOT / f"{args.dataset}.parquet"
        if args.dataset_path is None
        else args.dataset_path
    )
    df = pd.read_parquet(dataset_path)

    scorer = RewardScorer()
    all_scored_list: list[pd.DataFrame] = []
    for i in range(5):
        # Uses pre-sampled generation conditions from the test set unless sample=True
        generation_conditions = get_generation_conditions(df.iloc[i], sample=False)
        print("Prompt:")
        print(generation_conditions.to_prompt())
        print("-" * 100)
        generations = generate_molecules(generation_conditions)
        scored = scorer(generations, generation_conditions, parallel=False)
        all_scored_list.append(scored)

    all_scored = pd.concat(all_scored_list, ignore_index=True)
    write_success_stats(
        all_scored,
        Path(args.results_path) / "success_stats.json",
        dataset=args.dataset,
        condition=args.condition,
        reference_root=args.reference_root,
    )
    write_generation_results(all_scored, Path(args.results_path) / "scored_molecules.parquet")

if __name__ == "__main__":
    main()
