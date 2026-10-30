# Paper test datasets

This directory contains the exact clustered test datasets used for the 3D-Fit
paper results:

- `crossdocked.parquet`
- `plinder.parquet`

Each parquet contains the ligand SDF, receptor pocket PDB, and the pre-sampled
anchor-point, pharmacophore, and pocket-ligand-interaction conditions used in
the paper. These checked-in files are the benchmark's canonical test sets;
there is no separate full-versus-reduced dataset distinction.

Repository scripts use this directory by default. In particular,
`scripts/reduce_test_sets.py` overwrites the selected parquet in place when
reproducing the receptor-clustering procedure.
