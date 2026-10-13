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


def generate_molecules(generation_conditions: GenerationConditions) -> list[Chem.Mol]:
    """
    Call your generative model (LLM, diffusion, external binary, ...).

    Use any/all of the information in the generation conditions to generate molecules.
    Conditions provided in the generation conditions are used to score the molecules.

    Examples:
    - For an LLM, can do something like:

    generations = your_model(generation_conditions.to_prompt())
    
    and then pass the completions to the RewardScorer, which can parse the completions
    if the format instructions were followed by the modeland score the molecules.

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
            "Example usage of the 3D-Fit Benchmark library to score generated molecules."
            "The placeholder `generate_molecules` function is used to generate molecules based on the generation conditions."
            "in the example, the function returns a list of 2 copies of the seed molecule from the generation conditions."
        )
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to the test set parquet file.",
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
        default=None,
        help=(
            "Dataset name (crossdocked or plinder). Together with --condition, "
            "writes joint success, diversity, uniqueness, and novelty."
        ),
    )
    parser.add_argument(
        "--condition",
        type=str,
        default=None,
        help=(
            "Condition token (all_conditions, pocket, pocket_fragment, "
            "pocket_pharmacophore, or pocket_interaction). Together with "
            "--dataset, writes the set-level aggregate scores."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    df = pd.read_parquet(args.dataset_path)

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
    )
    write_generation_results(all_scored, Path(args.results_path) / "scored_molecules.parquet")

if __name__ == "__main__":
    main()
