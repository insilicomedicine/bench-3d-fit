"""
Top-level benchmark helpers (see ``examples/usage.py`` for a full walkthrough).

Typical flow:

1. :func:`get_generation_conditions` loads (or samples)
   :class:`~bench_3d_fit.generation_conditions.GenerationConditions` from a
   test-set row containing ``sdf``, ``pdb``, and optionally pre-sampled
   condition columns (``anchor_points``, ``pharmacophore_points``, ``pli_points``).
2. A :class:`~bench_3d_fit.reward.scorer.RewardScorer` scores a list of
   molecules against those conditions (``scorer(mols, conditions)``).
3. :func:`write_generation_results` writes scored Parquet / CSV and
   :func:`write_success_stats` writes aggregated metrics to JSON.
"""

from __future__ import annotations
from pathlib import Path
from typing import Union
import json
import numpy as np
import pandas as pd

from bench_3d_fit.sampling import sample_generation_conditions
from bench_3d_fit.generation_conditions import GenerationConditions
from bench_3d_fit.mol_io import safe_mol_to_mol_block, safe_mol_to_smiles

# ---------------------------------------------------------------------------
# Public API Functions
# ---------------------------------------------------------------------------

def get_generation_conditions(
    test_set_item: Union[dict, pd.Series],
    sample: bool = False,
) -> GenerationConditions:
    """
    Get generation conditions for a test set item. Can either be:
    1) sampled from the test set item
    2) pre-sampled using the sample_dataset_conditions.py script
    """
    if not sample and "anchor_points" not in test_set_item.keys():
        raise ValueError("Sampling is disabled but pre-sampled generation conditions (anchor_points, pharmacophore_points, pli_points) not found in test set item")
    if sample:
        conditions = sample_generation_conditions(test_set_item)
    else:
        conditions = GenerationConditions.from_test_row(test_set_item)
    return conditions


def write_generation_results(
    df: pd.DataFrame,
    path: Union[str, Path],
) -> None:
    """
    Write generated molecules and their scores to Parquet or CSV.

    Automatically adds SMILES column and converts RDKit molecules to MolBlock
    """
    path = Path(path)
    out = df.copy()
    suffix = path.suffix.lower()
    if suffix not in (".parquet", ".csv"):
        raise ValueError(
            f"Unsupported output suffix {path.suffix!r}; use .parquet or .csv"
        )
    
    if "ROMol" in df.columns:
        out["SMILES"] = out["ROMol"].apply(safe_mol_to_smiles)
        out["ROMol"] = out["ROMol"].apply(safe_mol_to_mol_block)
    path.parent.mkdir(parents=True, exist_ok=True)
    if suffix == ".parquet":
        out.to_parquet(path, index=False)
    elif suffix == ".csv":
        out.to_csv(path, index=False)


def write_success_stats(df: pd.DataFrame, path: Union[str, Path]) -> dict:
    """
    Write success stats of a single experiment to a JSON file.

    Columns ending in '_passed' are considered success rates, and are calculated on all inputs.
    (Nones are treated as 0s)
    Columns not ending in '_passed' are considered other scores, and are calculated on valid inputs.
    (Nones are treated as NaNs and ignored)

    To analyze multiple experiments:
        Stack these single score dictionaries into a single DataFrame with something like:
        pandas.DataFrame.from_dict([write_success_stats(df, path) for path in paths])
    """
    path = Path(path)
    success_columns = [c for c in df.columns if c.endswith('_passed')]
    success_df = df[success_columns]
    success_df = success_df.fillna(0)
    success_rates = success_df.values.astype(int)
    success_rates = np.mean(success_rates, axis=0)
    other_columns = [c for c in df.columns if c not in success_columns + ['SMILES', 'ROMol']]
    other_df = df[other_columns]
    other_scores = other_df.values.astype(float)
    other_scores = np.nanmean(other_scores, axis=0)
    scores = {
        k: v for k, v in zip(
            success_columns + other_columns,
            success_rates.tolist() + other_scores.tolist()
        )
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(scores, f)