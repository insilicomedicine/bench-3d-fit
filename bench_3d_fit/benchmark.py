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
   Column means are always included. When ``dataset`` and ``condition`` are
   passed and ``aggregate_scores`` is true, set-level scores (joint success,
   diversity, uniqueness, novelty) are merged in as well.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Union
import json
import warnings
import numpy as np
import pandas as pd

from bench_3d_fit.sampling import sample_generation_conditions
from bench_3d_fit.generation_conditions import GenerationConditions
from bench_3d_fit.mol_io import safe_mol_to_mol_block, safe_mol_to_smiles
from bench_3d_fit.reward.aggregate import compute_aggregate_scores

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
    
    for c in df.columns:
        if "ROMol" in c:
            out[c] = out[c].apply(safe_mol_to_mol_block)
        if "SMILES" in c:
            out[c] = out[c].apply(safe_mol_to_smiles)
    path.parent.mkdir(parents=True, exist_ok=True)
    if suffix == ".parquet":
        out.to_parquet(path, index=False)
    elif suffix == ".csv":
        out.to_csv(path, index=False)


DEFAULT_STAT_AGGREGATIONS = {
    'mean': [
        'valid_mol_passed', 'posebusters_intramolecular_passed',
        'posebusters_internal_energy_passed', 'posebusters_intermolecular_passed',
        'unidock_passed', 'anchor_points_2d_passed', 'anchor_points_3d_passed',
        'pharmacophore_passed', 'pocket_ligand_interaction_passed',
        'descriptors_ro5_passed', 'descriptors_ro5_mw_passed',
        'descriptors_ro5_hbd_passed', 'descriptors_ro5_hba_passed',
        'descriptors_ro5_logp_passed',
        'ligand_similarity_fp_passed', 'ligand_similarity_shape_passed',
        'ligand_similarity_esp_passed',
        'opt_ligand_similarity_fp_passed', 'opt_ligand_similarity_shape_passed',
        'opt_ligand_similarity_esp_passed',
        'opt_posebusters_intramolecular_passed',
        'opt_posebusters_internal_energy_passed', 'opt_posebusters_intermolecular_passed',
        'opt_unidock_passed', 'opt_anchor_points_3d_passed',
        'opt_pharmacophore_passed', 'opt_pocket_ligand_interaction_passed',
    ],
    'nanmedian': ['unidock_score', 'opt_unidock_score'],
    'nanmean': [
        'anchor_points_3d_atom_frac', 'pharmacophore_rms',
        'per_bond_mmff_energy_score', 'ligand_similarity_fp',
        'ligand_similarity_shape', 'ligand_similarity_esp', 'descriptors_nheavy',
        'descriptors_mw', 'descriptors_hbd', 'descriptors_hba', 'descriptors_logp',
        'descriptors_tpsa', 'descriptors_qed', 
    ],
}


def get_success_stats(
    df: pd.DataFrame,
    stat_aggregations: dict = None,
    dataset: Optional[str] = None,
    condition: Optional[str] = None,
    reference_root: Optional[Union[str, Path]] = None,
    aggregate_scores: bool = True,
) -> dict:
    """
    Get success stats of a dataframe.

    Columns ending in '_passed' are considered success rates, and are calculated on all inputs.
    (Nones are treated as 0s)
    Columns not ending in '_passed' are considered other scores, and are calculated on valid inputs.
    (Nones are treated as NaNs and ignored)

    When ``aggregate_scores`` is true and both ``dataset`` and ``condition``
    are provided, set-level scores are merged in: joint condition success
    (using ``condition``), Morgan diversity, uniqueness, and novelty against
    the dataset reference split. Those scores need the full frame and are not
    means of per-molecule reward columns. Set ``aggregate_scores`` to false to
    skip them when they are not needed. Novelty uses
    ``data/novelty_reference/{dataset}_unique.txt`` from the repository by
    default. ``reference_root`` or ``BENCH_3D_FIT_REFERENCE_ROOT`` can
    override that directory.

    Returns a dictionary with the success rates (all outputs) and other scores (valid inputs only).
    """
    if stat_aggregations is None:
        stat_aggregations = DEFAULT_STAT_AGGREGATIONS
    all_scores = {}
    for stat_type, all_columns in stat_aggregations.items():
        columns = [c for c in all_columns if c in df.columns]
        if len(columns) == 0:
            continue
        if stat_type == 'mean':
            sub_df = df[columns]
            sub_df = sub_df.fillna(0)
            scores = sub_df.values.astype(int)
            scores = np.mean(scores, axis=0)
            for col, value in zip(columns, scores):
                all_scores[col] = value
        elif stat_type == 'nanmedian':
            sub_df = df[columns]
            sub_df = sub_df.fillna(np.nan)
            scores = sub_df.values.astype(float)
            scores = np.nanmedian(scores, axis=0)
            for col, value in zip(columns, scores):
                all_scores[col] = value
        elif stat_type == 'nanmean':
            sub_df = df[columns]
            sub_df = sub_df.fillna(np.nan)
            scores = sub_df.values.astype(float)
            scores = np.nanmean(scores, axis=0)
            for col, value in zip(columns, scores):
                all_scores[col] = value

    if aggregate_scores and (dataset is None) != (condition is None):
        warnings.warn(
            "Aggregate scores require both dataset and condition; "
            f"got dataset={dataset!r} condition={condition!r}. "
            "Skipping aggregate scores.",
            stacklevel=2,
        )
    elif aggregate_scores and dataset is not None and condition is not None:
        all_scores.update(
            compute_aggregate_scores(
                df,
                dataset,
                condition,
                reference_root=reference_root,
            )
        )
    return all_scores


def write_success_stats(
    df: pd.DataFrame,
    path: Union[str, Path],
    dataset: Optional[str] = None,
    condition: Optional[str] = None,
    reference_root: Optional[Union[str, Path]] = None,
    aggregate_scores: bool = True,
) -> dict:
    """
    Get and write success stats of a single experiment to a JSON file.

    ``dataset``, ``condition``, and ``aggregate_scores`` are forwarded to
    :func:`get_success_stats`. Joint success, diversity, uniqueness, and
    novelty are included when both dataset and condition are set and
    ``aggregate_scores`` is true. Novelty uses the bundled paper reference
    SMILES by default; ``reference_root`` or
    ``BENCH_3D_FIT_REFERENCE_ROOT`` can override their directory.

    Returns the scores dictionary.
    """
    path = Path(path)
    scores = get_success_stats(
        df,
        dataset=dataset,
        condition=condition,
        reference_root=reference_root,
        aggregate_scores=aggregate_scores,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(scores, f)