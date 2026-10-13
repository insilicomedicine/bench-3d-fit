# 3D-Fit Benchmark
Benchmarking Code for: Do Language Models Dream of Binding Molecules? Benchmarking LLMs under Spatial Constraints

Scoring and evaluation framework for 3D **generative chemistry** models with 3D conditions.  Your generator stays outside the package — you pass in RDKit molecules (or LLM completion strings according to the included prompts) and get back scored DataFrames for your molecules (Parquet / CSV) plus aggregated metrics.

## Quick start

See `examples/usage.py` for a complete walkthrough.

```python
from pathlib import Path
import pandas as pd
from bench_3d_fit import (
    RewardScorer,
    GenerationConditions,
    get_generation_conditions,
    write_generation_results,
    write_success_stats,
)

df = pd.read_parquet("test.parquet")
scorer = RewardScorer()          # uses all registered reward functions
all_scored = []

for i in range(min(5, len(df))):
    conditions = get_generation_conditions(df.iloc[i], sample=False)
    generations = your_model(conditions)        # list[Chem.Mol] or list[str]
    scored = scorer(generations, conditions)
    all_scored.append(scored)

results = pd.concat(all_scored, ignore_index=True)
write_generation_results(results, Path("output/scored_molecules.parquet"))
write_success_stats(results, Path("output/success_stats.json"))
```

### LLM usage

For instruction-following LLMs, build a prompt directly from the conditions:

```python
prompt = conditions.to_prompt(
    conditions_to_include=["pocket", "anchor_points", "pharmacophore", "pocket_ligand_interaction"],
    format_instructions="sdf_enum",
)
completions = your_llm(prompt)   # list[str]
scored = scorer(completions, conditions)
```

The `RewardScorer` automatically parses `<sdf>` / `<smiles_xyz>` tagged completion strings.

### Diffusion / specialist models

For non-LLM generators, access the raw condition fields:

```python
env_pdb = conditions.environment_pdb
anchors = conditions.anchor_points
# Process the conditions for input to the model
molecules = your_diffusion_model(env_pdb, anchors)   # list[Chem.Mol]
scored = scorer(molecules, conditions)
```

## Installation

```bash
git clone <repository-url>
cd bench-3d-fit
pip install -e .
```

### UniDock

The UniDock docking reward requires the `unidock` binary on `PATH`. We chose to keep this
out of the main dependencies of the project, so all dependencies can remain pip installable
for ease of installation of the benchmark on top of existing molecular generator environments.

If you already have a conda environment you can install it directly:

```bash
conda install -c conda-forge unidock
```

Otherwise, install it in a separate environment and symlink:

```bash
yes | "${SHELL}" <(curl -L micro.mamba.pm/install.sh)
micromamba create -n unidock -c conda-forge unidock -y
ln -s "$(micromamba run -n unidock which unidock)" /usr/local/bin/unidock
```

You can also set the `UNIDOCK_BIN` environment variable to the full path of the binary.

## Dataset preparation

### 1. Process raw data

Convert CrossDocked2020 or PLInder raw files into a cleaned parquet:

```bash
python scripts/process_raw_data.py \
    --dataset crossdocked \
    --input-dir /path/to/crossdocked/ \
    --split-file /path/to/test0.types \
    --output-dir /path/to/output/
```

### 2. Pre-sample generation conditions

Augment the parquet with sampled anchor points, pharmacophore points, and pocket-ligand interaction points:

```bash
python scripts/sample_dataset_conditions.py \
    --input-file /path/to/test.parquet \
    --output-file /path/to/test_with_conditions.parquet
```

## Key components

| Component | Role |
|-----------|------|
| `GenerationConditions` | Dataclass carrying all generation context: pocket PDB text, seed molecule, anchor points, pharmacophore points, pocket-ligand interactions |
| `RewardScorer` | Runs registered `RewardFunction` instances (optionally in parallel) and returns a scored DataFrame |
| `RewardFunction` | Abstract base class for one scoring criterion; subclass and override `compute(mols, conditions)` |
| `register_reward` / `get_reward_class` | Name-to-class registry (core rewards are auto-registered) |
| `get_generation_conditions` | Load or sample `GenerationConditions` from a test-set row |
| `sample_generation_conditions` | Sample `GenerationConditions` from an annotated parquet row at runtime |
| `write_generation_results` | Write scored molecules to Parquet or CSV |
| `get_success_stats` / `write_success_stats` | Aggregate pass rates and scores, including raw vs optimized-pose columns |

`GenerationConditions.to_prompt` defaults to the `druglike_ro5` molecule conditions (Lipinski rule of five, 20–40 heavy atoms). Other variants are `base`, `druglike_nheavy_15`, `druglike_nheavy_25`, and `druglike_nheavy_seed`.

`RewardScorer` defaults to `rescore_optimized_poses=True`. It scores the parsed pose (UniDock `--score_only`, columns without a prefix) and the UniDock locally optimized pose (columns prefixed with `opt_`).

## Built-in reward functions

Registered in `bench_3d_fit.reward.rewards`:

| Name | Class | Scores |
|------|-------|--------|
| `descriptors` | `DescriptorsReward` | heavy-atom count, MW, HBD, HBA, logP, TPSA, QED, rule-of-five pass flags |
| `posebusters` | `PoseBustersReward` | `intramolecular_passed`, `intermolecular_passed` |
| `unidock` | `UniDockReward` | `passed`, `score` (kcal/mol), `ROMol` |
| `anchor_points` | `AnchorPointsReward` | `2d_passed`, `3d_passed`, `3d_atom_frac` |
| `pharmacophore` | `PharmacophoreReward` | `passed`, `rms` |
| `pocket_ligand_interaction` | `PLIPointsReward` | `passed` |
| `ligand_similarity` | `LigandSimilarityReward` | `fp`, `shape`, `esp` and matching `*_passed` flags |

## Project layout

```
bench-3d-fit/
├── README.md
├── LICENSE
├── pyproject.toml
├── configs/
│   └── posebusters.yml              # PoseBusters configuration
├── scripts/
│   ├── process_raw_data.py          # Process CrossDocked2020 / PLInder raw data
│   └── sample_dataset_conditions.py # Pre-sample conditions for a parquet test set
├── examples/
│   └── usage.py                     # End-to-end scoring example
└── bench_3d_fit/
    ├── __init__.py                  # Clean public API
    ├── benchmark.py                 # get_generation_conditions, write helpers
    ├── generation_conditions.py     # GenerationConditions dataclass + from_test_row
    ├── mol_io.py                    # Molecule I/O (parsing, reconstruction, validation, coords)
    ├── prompt.py                    # LLM prompt construction (format_environment_pdb, etc.)
    ├── sampling.py                  # sample_generation_conditions + BRICS fragment helpers
    └── reward/
        ├── __init__.py              # Re-exports RewardScorer, RewardFunction, registry
        ├── base.py                  # RewardFunction ABC + registry (register_reward / get_reward_class)
        ├── scorer.py                # RewardScorer (molecule parsing + multi-reward runner)
        ├── rewards.py               # Built-in RewardFunction subclasses
        ├── aggregate.py             # Set-level joint success, diversity, uniqueness, novelty
        └── utils.py                 # PDB-to-PDBQT, pharmacophore, PLI helpers
```

## Requirements

Python >= 3.9; RDKit, NumPy, pandas, SciPy, espsim, pmapper, ProDy, ProLIF, PoseBusters, UniDock (installed separately — see above).
