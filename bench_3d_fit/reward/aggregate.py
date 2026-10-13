"""
Set-level scores that cannot be reduced from per-molecule reward columns.

These run once on a fully scored frame, after
:class:`~bench_3d_fit.reward.scorer.RewardScorer`. They need context the
per-molecule scorer does not have: the condition token for the task, the
dataset name, and (for novelty) the train+test reference SMILES.

* ``joint_conditions_passed`` / ``opt_joint_conditions_passed`` — mean of a
  row-wise AND over the pass columns that belong to the condition token.
* ``diversity`` — ``1 - mean pairwise Morgan Tanimoto`` over valid molecules.
* ``uniqueness`` — unique heavy-atom canonical SMILES divided by valid molecules.
* ``novelty`` — fraction of valid molecules absent from the train+test reference.

Diversity and uniqueness are not valid on a single prompt group: duplicates
and pairwise similarities do not average across batches. Novelty's denominator
is valid molecules only, so it is not a ``*_passed`` column.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

from bench_3d_fit.mol_io import mol_from_sdf_cell

logger = logging.getLogger(__name__)
RDLogger.DisableLog("rdApp.*")

# Morgan fingerprint settings for diversity (matches common psi / genchem usage).
_DIVERSITY_MORGAN_RADIUS = 2
_DIVERSITY_FP_SIZE = 2048
_DIVERSITY_FP_GENERATOR = AllChem.GetMorganGenerator(
    radius=_DIVERSITY_MORGAN_RADIUS,
    fpSize=_DIVERSITY_FP_SIZE,
)

# Default location of ``genchembench_{dataset}/v2/test.parquet``. Callers can
# override this per computation; it is not fixed inside the loaders.
DEFAULT_REFERENCE_ROOT = Path(
    "/mnt/custom-file-systems/efs/fs-05a219e38ecb2cb6f/data"
)
TEST_PARQUET_TEMPLATE = "genchembench_{dataset}/v2/test.parquet"

# Train+test unique heavy-atom canonical SMILES, one per line.
# ``reference_root`` stays the full v2 test parquet root; these lists live
# with the reduced test data.
UNIQUE_SMILES_ROOT = Path(
    "/mnt/custom-file-systems/efs/fs-05a219e38ecb2cb6f/t.macdougall/dev/"
    "experiments/2026_10_genchem/3dfit_resubmission/reduced_v2_test_data"
)
UNIQUE_SMILES_TEMPLATE = "genchembench_{dataset}/v2/{dataset}_unique.txt"

TASK_NAME_PATTERN = re.compile(
    r"(mini_test|crossdocked|plinder)_"
    r"(all_conditions|pocket|pocket_pharmacophore|pocket_fragment|pocket_interaction)_"
    r"(xyz|smiles_xyz|sdf_enum|sdf|smiles_enum|smiles_edits|diff)"
)

ScoreAdder = Callable[..., Dict[str, float]]

# Base filters that always enter the joint AND.
_JOINT_BASE_COLUMNS = [
    "descriptors_ro5_passed",
    "unidock_passed",
    "posebusters_intermolecular_passed",
    "posebusters_intramolecular_passed",
]

# Extra condition-specific filters (raw / non-opt names).
_JOINT_EXTRA_BY_CONDITION: Dict[str, List[str]] = {
    "pocket": [],
    "pocket_fragment": ["anchor_points_3d_passed"],
    "pocket_pharmacophore": ["pharmacophore_passed"],
    "pocket_interaction": ["pocket_ligand_interaction_passed"],
    "all_conditions": ["anchor_points_3d_passed", "pharmacophore_passed",
                       "pocket_ligand_interaction_passed"],
}

# RO5 has no separate opt column for joint scoring; reuse the raw column.
_NO_OPT_PREFIX_COLUMNS = {"descriptors_ro5_passed"}


def parse_task(task: Union[str, Path]) -> Tuple[str, str]:
    """Extract ``(dataset, condition)`` from a task name or experiment dirname."""
    name = task.name if isinstance(task, Path) else Path(task).name
    match = TASK_NAME_PATTERN.fullmatch(name)
    if match is None:
        raise ValueError(
            f"Could not parse dataset/condition from task name {name!r}. "
            f"Expected pattern like 'crossdocked_all_conditions_sdf_enum'."
        )
    return match.group(1), match.group(2)


def joint_filter_columns(condition: str, *, opt: bool = False) -> List[str]:
    """Return the columns that enter the joint success AND."""
    if condition not in _JOINT_EXTRA_BY_CONDITION:
        raise ValueError(
            f"Unknown condition {condition!r}. Expected one of "
            f"{sorted(_JOINT_EXTRA_BY_CONDITION)}."
        )
    raw_cols = _JOINT_BASE_COLUMNS + _JOINT_EXTRA_BY_CONDITION[condition]
    if not opt:
        return list(raw_cols)

    opt_cols: List[str] = []
    for col in raw_cols:
        if col in _NO_OPT_PREFIX_COLUMNS:
            opt_cols.append(col)
        else:
            opt_cols.append(f"opt_{col}")
    return opt_cols


def _joint_pass_rate(df: pd.DataFrame, columns: Sequence[str]) -> float:
    """Mean of per-row AND across ``columns``; missing/None treated as fail."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"Scored frame is missing required columns for joint success: "
            f"{missing}. Present columns: {sorted(df.columns)}."
        )
    sub = df.loc[:, list(columns)].fillna(0)
    passed = sub.astype(bool).all(axis=1).to_numpy(dtype=np.float64)
    if passed.size == 0:
        return float("nan")
    return float(np.mean(passed))


def add_joint_conditions_passed(
    df: pd.DataFrame,
    condition: str,
    dataset: str,
    *,
    reference_root: Optional[Union[str, Path]] = None,
) -> Dict[str, float]:
    """Compute joint success rates across condition-relevant pass filters."""
    del dataset, reference_root
    raw_cols = joint_filter_columns(condition, opt=False)
    opt_cols = joint_filter_columns(condition, opt=True)
    return {
        "joint_conditions_passed": _joint_pass_rate(df, raw_cols),
        "opt_joint_conditions_passed": _joint_pass_rate(df, opt_cols),
    }


def _canonical_smiles(mol: Chem.Mol) -> Optional[str]:
    """Heavy-atom canonical SMILES for identity comparisons."""
    try:
        return Chem.MolToSmiles(Chem.RemoveHs(mol))
    except Exception:
        return None


def _require_romol(df: pd.DataFrame, purpose: str) -> None:
    if "ROMol" not in df.columns:
        raise ValueError(
            "Scored frame is missing required column 'ROMol' "
            f"for {purpose}. Present columns: {sorted(df.columns)}."
        )


def _mols_from_cells(values) -> List[Chem.Mol]:
    """Parse cells with :func:`mol_from_sdf_cell`, skipping invalid ones."""
    mols: List[Chem.Mol] = []
    for value in values:
        mol = mol_from_sdf_cell(value)
        if mol is not None:
            mols.append(mol)
    return mols


def _valid_mols(df: pd.DataFrame, purpose: str) -> List[Chem.Mol]:
    """Parse the ``ROMol`` column, skipping invalid or empty cells."""
    _require_romol(df, purpose)
    return _mols_from_cells(df["ROMol"])


def _canonical_smiles_list(mols: Sequence[Chem.Mol]) -> List[str]:
    """Heavy-atom canonical SMILES, dropping molecules that fail conversion."""
    smiles: List[str] = []
    for mol in mols:
        smi = _canonical_smiles(mol)
        if smi:
            smiles.append(smi)
    return smiles


def _valid_canonical_smiles(df: pd.DataFrame, purpose: str) -> List[str]:
    """Canonical SMILES for each parseable ``ROMol`` in a scored frame."""
    return _canonical_smiles_list(_valid_mols(df, purpose))


def _rate(numerator: int, denominator: int) -> float:
    """``numerator / denominator``, or NaN when the denominator is empty."""
    if denominator == 0:
        return float("nan")
    return float(numerator / denominator)


def _pairwise_morgan_diversity(mols: Sequence[Chem.Mol]) -> float:
    """
    Diversity = 1 - mean pairwise Morgan Tanimoto similarity.

    Requires at least two molecules; otherwise returns NaN.
    """
    if len(mols) < 2:
        return float("nan")

    fps = [_DIVERSITY_FP_GENERATOR.GetFingerprint(mol) for mol in mols]
    pairwise_sims: List[float] = []
    for i, fp in enumerate(fps[:-1]):
        pairwise_sims.extend(DataStructs.BulkTanimotoSimilarity(fp, fps[i + 1 :]))
    if not pairwise_sims:
        return float("nan")
    return float(1.0 - np.mean(pairwise_sims))


def add_diversity(
    df: pd.DataFrame,
    condition: str,
    dataset: str,
    *,
    reference_root: Optional[Union[str, Path]] = None,
) -> Dict[str, float]:
    """
    RDKit Morgan-fingerprint diversity over all valid generated molecules.

    Uses radius-2 / 2048-bit fingerprints. Invalid / empty ``ROMol`` entries
    are skipped. Accepts in-memory ``Chem.Mol`` objects and MolBlock strings.
    """
    del condition, dataset, reference_root
    mols = _valid_mols(df, "diversity")
    diversity = _pairwise_morgan_diversity(mols)
    logger.info(
        "Diversity computed over %d / %d molecules",
        len(mols),
        len(df),
    )
    return {"diversity": diversity}


@lru_cache(maxsize=4)
def _load_reference_smiles(dataset: str) -> Set[str]:
    """Load train+test canonical SMILES for one dataset."""
    if dataset not in ("crossdocked", "plinder"):
        raise ValueError(
            f"Novelty requires dataset 'crossdocked' or 'plinder'; got {dataset!r}."
        )
    path = UNIQUE_SMILES_ROOT / UNIQUE_SMILES_TEMPLATE.format(dataset=dataset)
    if not path.is_file():
        raise FileNotFoundError(f"Reference SMILES file not found: {path}")

    smiles = {line for line in path.read_text().splitlines() if line}
    logger.info("Loaded %d reference SMILES from %s", len(smiles), path)
    return smiles


def add_novelty(
    df: pd.DataFrame,
    condition: str,
    dataset: str,
    *,
    reference_root: Optional[Union[str, Path]] = None,
) -> Dict[str, float]:
    """
    Fraction of valid generated molecules absent from the train+test reference.

    Reference SMILES are the heavy-atom canonical SMILES in
    ``genchembench_{dataset}/v2/{dataset}_unique.txt`` under
    :data:`UNIQUE_SMILES_ROOT`. The denominator is valid molecules only.
    """
    del condition, reference_root
    smiles = _valid_canonical_smiles(df, "novelty")
    reference = _load_reference_smiles(dataset)
    n_valid = len(smiles)
    n_novel = sum(smi not in reference for smi in smiles)
    novelty = _rate(n_novel, n_valid)
    logger.info(
        "Novelty: %d / %d valid molecules not in %s references",
        n_novel,
        n_valid,
        dataset,
    )
    return {"novelty": novelty}


def add_uniqueness(
    df: pd.DataFrame,
    condition: str,
    dataset: str,
    *,
    reference_root: Optional[Union[str, Path]] = None,
) -> Dict[str, float]:
    """
    Fraction of valid generated molecules that are unique from each other.

    uniqueness = |unique canonical SMILES| / |valid molecules|.
    """
    del condition, dataset, reference_root
    smiles = _valid_canonical_smiles(df, "uniqueness")
    n_valid = len(smiles)
    n_unique = len(set(smiles))
    uniqueness = _rate(n_unique, n_valid)
    logger.info(
        "Uniqueness: %d unique / %d valid molecules",
        n_unique,
        n_valid,
    )
    return {"uniqueness": uniqueness}


SCORE_ADDERS: Dict[str, ScoreAdder] = {
    "joint_conditions": add_joint_conditions_passed,
    "diversity": add_diversity,
    "novelty": add_novelty,
    "uniqueness": add_uniqueness,
}
AGGREGATE_SCORE_NAMES: Tuple[str, ...] = tuple(SCORE_ADDERS)


def compute_aggregate_scores(
    df: pd.DataFrame,
    dataset: str,
    condition: str,
    *,
    reference_root: Optional[Union[str, Path]] = None,
    scores: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    """
    Compute set-level scores for one scored experiment frame.

    ``scores`` selects adder names (default: all of
    :data:`AGGREGATE_SCORE_NAMES`). Novelty reads train+test SMILES from
    :data:`UNIQUE_SMILES_ROOT` and does not use ``reference_root``.
    """
    adder_names = list(AGGREGATE_SCORE_NAMES if scores is None else scores)
    unknown = [name for name in adder_names if name not in SCORE_ADDERS]
    if unknown:
        raise ValueError(
            f"Unknown score adder(s): {unknown}. "
            f"Available: {sorted(SCORE_ADDERS)}"
        )

    new_scores: Dict[str, float] = {}
    for name in adder_names:
        adder_scores = SCORE_ADDERS[name](
            df,
            condition,
            dataset,
            reference_root=reference_root,
        )
        overlap = set(new_scores) & set(adder_scores)
        if overlap:
            raise ValueError(
                f"Score adder {name!r} produced keys already emitted by "
                f"another adder in this run: {sorted(overlap)}"
            )
        new_scores.update(adder_scores)
    return new_scores
