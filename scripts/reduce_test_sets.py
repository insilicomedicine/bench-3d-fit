"""Cluster CrossDocked and PLINDER test sets in place.

Keeps the first row for each heavy-atom canonical SMILES, then the first
remaining row for each exact receptor sequence. The remaining rows are
clustered by longest-chain sequence identity, and each cluster keeps its
first row in dataset order. Condition columns stay on the kept rows. A
sidecar JSON lists the 0-based positions of those rows in the input parquet.
Each source parquet is atomically replaced with its clustered result.
Paths default to the checked-in ``data/test_sets/{dataset}.parquet`` files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd
from Bio.Align import PairwiseAligner
from rdkit import Chem

from bench_3d_fit.mol_io import mol_from_sdf_cell
from bench_3d_fit.reward.aggregate import (
    DEFAULT_TEST_DATA_ROOT,
    TEST_DATA_ROOT_ENV,
)

DATASETS = ("crossdocked", "plinder")
TEST_PARQUET_TEMPLATE = "{dataset}.parquet"
# Global identity on the longest chain. Below this, a row starts a new cluster.
SEQUENCE_IDENTITY = 0.50

# Standard residues only. Anything else, including MSE, becomes X.
_AMINO_ACIDS = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Drop repeated ligands and receptor sequences, then keep the "
            "first row of each protein-sequence cluster."
        )
    )
    parser.add_argument(
        "--test-data-root",
        "--reference-root",
        dest="test_data_root",
        type=Path,
        default=DEFAULT_TEST_DATA_ROOT,
        help=(
            "Directory containing crossdocked.parquet and plinder.parquet. "
            f"Defaults to ${TEST_DATA_ROOT_ENV} when set, otherwise "
            "data/test_sets/."
        ),
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASETS),
        choices=DATASETS,
        help="Datasets to reduce. Defaults to crossdocked and plinder.",
    )
    return parser.parse_args()


def canonical_smiles(sdf_cell) -> Optional[str]:
    """Heavy-atom canonical SMILES, or None when the ligand does not parse."""
    mol = mol_from_sdf_cell(sdf_cell)
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(Chem.RemoveHs(mol))
    except Exception:
        return None


def receptor_sequence(pdb_text) -> Optional[str]:
    """Chain sequences joined in chain-id order, or None when no CA atoms exist.

    Each chain is the CA residues in file order. Standard residues map to
    one-letter codes; every other residue name is X. A multi-chain receptor
    matches another only when every chain sequence matches.
    """
    if pdb_text is None or (isinstance(pdb_text, float) and pd.isna(pdb_text)):
        return None
    chains: dict[str, list[str]] = {}
    seen: set[tuple[str, str]] = set()
    for line in str(pdb_text).splitlines():
        if not line.startswith("ATOM") or line[12:16].strip() != "CA":
            continue
        chain_id = line[21]
        residue_key = (chain_id, line[22:27])
        if residue_key in seen:
            continue
        seen.add(residue_key)
        residue = _AMINO_ACIDS.get(line[17:20].strip(), "X")
        chains.setdefault(chain_id, []).append(residue)
    if not chains:
        return None
    return "|".join("".join(chains[chain_id]) for chain_id in sorted(chains))


def longest_chain_sequence(pdb_text) -> Optional[str]:
    """Longest chain from :func:`receptor_sequence`, or None when it is missing."""
    signature = receptor_sequence(pdb_text)
    if signature is None:
        return None
    return max(signature.split("|"), key=len)


def _aligner() -> PairwiseAligner:
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 1.0
    aligner.mismatch_score = -1.0
    aligner.open_gap_score = -1.0
    aligner.extend_gap_score = -0.5
    return aligner


def sequence_identity(
    left: str,
    right: str,
    aligner: PairwiseAligner,
    threshold: float,
) -> float:
    """Identical columns over the global alignment length."""
    if len(left) > len(right):
        left, right = right, left
    if not right:
        return 1.0
    # Global identity cannot exceed the length ratio.
    if len(left) / len(right) < threshold:
        return len(left) / len(right)
    alignment = aligner.align(left, right)[0]
    return alignment.counts().identities / alignment.length


def cluster_by_sequence(
    sequences: Sequence[Optional[str]],
    threshold: float = SEQUENCE_IDENTITY,
) -> list[int]:
    """Assign rows to clusters in dataset order.

    A row joins the earliest cluster whose first member has global identity
    at or above ``threshold``. A missing sequence never joins another row.
    """
    aligner = _aligner()
    representatives: list[Optional[str]] = []
    labels: list[int] = []
    for sequence in sequences:
        if sequence is None:
            labels.append(len(representatives))
            representatives.append(None)
            continue
        assigned = None
        for cluster_id, representative in enumerate(representatives):
            if representative is None:
                continue
            if sequence_identity(sequence, representative, aligner, threshold) >= threshold:
                assigned = cluster_id
                break
        if assigned is None:
            assigned = len(representatives)
            representatives.append(sequence)
        labels.append(assigned)
    return labels


def first_occurrence_mask(keys: Sequence[Optional[str]]) -> list[bool]:
    """Keep the first row for each key. None never matches another row."""
    seen: set[str] = set()
    keep: list[bool] = []
    for key in keys:
        if key is None or key not in seen:
            keep.append(True)
            if key is not None:
                seen.add(key)
        else:
            keep.append(False)
    return keep


def source_path(test_data_root: Path, dataset: str) -> Path:
    path = test_data_root / TEST_PARQUET_TEMPLATE.format(dataset=dataset)
    if path.name != f"{dataset}.parquet":
        raise ValueError(f"Unexpected test-set path: {path}")
    return path


def index_path(parquet_path: Path) -> Path:
    return parquet_path.with_name("kept_source_indices.json")


def apply_keep_mask(
    frame: pd.DataFrame,
    source_positions: list[int],
    keep: Sequence[bool],
) -> tuple[pd.DataFrame, list[int]]:
    """Return kept rows and their positions in the input parquet."""
    if len(frame) != len(source_positions) or len(frame) != len(keep):
        raise ValueError(
            "Row mask, frame, and source positions must have the same length."
        )
    kept_rows = [index for index, flag in enumerate(keep) if flag]
    kept_positions = [source_positions[index] for index in kept_rows]
    kept_frame = frame.iloc[kept_rows].reset_index(drop=True)
    return kept_frame, kept_positions


def reduce_dataset(test_data_root: Path, dataset: str) -> None:
    src = source_path(test_data_root, dataset)
    if not src.is_file():
        raise FileNotFoundError(f"Test set not found: {src}")

    df = pd.read_parquet(src)
    for column in ("sdf", "pdb", "anchor_points", "pharmacophore_points", "pli_points"):
        if column not in df.columns:
            raise ValueError(
                f"{src} is missing required column {column!r}. "
                f"Present columns: {list(df.columns)}"
            )

    n_start = len(df)
    source_positions = list(range(n_start))
    smiles_keep = first_occurrence_mask([canonical_smiles(value) for value in df["sdf"]])
    after_smiles, source_positions = apply_keep_mask(df, source_positions, smiles_keep)
    n_after_smiles = len(after_smiles)

    sequence_keep = first_occurrence_mask(
        [receptor_sequence(value) for value in after_smiles["pdb"]]
    )
    after_sequence, source_positions = apply_keep_mask(
        after_smiles, source_positions, sequence_keep
    )
    n_after_sequence = len(after_sequence)

    labels = cluster_by_sequence(
        [longest_chain_sequence(value) for value in after_sequence["pdb"]]
    )
    cluster_keep = first_occurrence_mask([str(label) for label in labels])
    reduced, kept_indices = apply_keep_mask(
        after_sequence, source_positions, cluster_keep
    )

    indices_dest = index_path(src)
    parquet_tmp = src.with_name(f".{src.name}.tmp")
    indices_tmp = indices_dest.with_name(f".{indices_dest.name}.tmp")
    try:
        parquet_tmp.unlink(missing_ok=True)
        indices_tmp.unlink(missing_ok=True)
        reduced.to_parquet(parquet_tmp, index=False)
        indices_tmp.write_text(
            json.dumps(
                {
                    "dataset": dataset,
                    "source": str(src),
                    "n_source_rows": n_start,
                    "kept_source_indices": kept_indices,
                },
                indent=2,
            )
            + "\n"
        )
        parquet_tmp.replace(src)
        indices_tmp.replace(indices_dest)
    finally:
        parquet_tmp.unlink(missing_ok=True)
        indices_tmp.unlink(missing_ok=True)

    print(
        f"{dataset}: read {src} ({n_start} rows), "
        f"SMILES pass kept {n_after_smiles} dropped {n_start - n_after_smiles}, "
        f"exact sequence pass kept {n_after_sequence} dropped {n_after_smiles - n_after_sequence}, "
        f"{len(set(labels))} sequence clusters at {SEQUENCE_IDENTITY:.0%} identity, "
        f"kept one representative {len(reduced)} dropped {n_after_sequence - len(reduced)}"
    )
    print(f"  overwrote {src}")
    print(f"  wrote {indices_dest} ({len(kept_indices)} source indices)")


def main() -> None:
    args = parse_args()
    for dataset in args.datasets:
        reduce_dataset(args.test_data_root, dataset)


if __name__ == "__main__":
    main()
