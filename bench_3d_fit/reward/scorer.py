"""
RewardScorer: multi-reward runner and molecule-string parser.

:class:`RewardScorer` manages a collection of :class:`RewardFunction` objects,
applies them to a batch of molecules, and combines the results into a
single scored :class:`~pandas.DataFrame`.

Constructed **once** with a list of registered reward names (default: all
registered rewards).  Each ``__call__`` receives molecules *and*
:class:`~bench_3d_fit.generation_conditions.GenerationConditions` (or one
per molecule) carrying context (seed molecule, receptor PDB, anchors,
pharmacophore points, pocket-ligand interactions).
"""

from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import logging
import re
from typing import List, Optional, Union

import pandas as pd
from rdkit import Chem

from bench_3d_fit.mol_io import (
    reconstruct_full_sdf,
    reconstruct_simple_sdf,
    reconstruct_smiles_enum,
    reconstruct_smiles_xyz,
)
from bench_3d_fit.generation_conditions import GenerationConditions
from bench_3d_fit.reward.base import (
    RewardFunction,
    get_reward_class,
    _REWARD_REGISTRY,
)
from bench_3d_fit.reward.rewards import UniDockReward


logger = logging.getLogger(__name__)

MoleculeType3D = Union[Chem.Mol, str]


class RewardScorer:
    """Container and runner for multiple reward functions.

    Constructed with an optional list of registered reward *names*; when
    omitted, every reward in the registry is included.

    Example::

        scorer = RewardScorer(["posebusters", "unidock"])
        conditions = GenerationConditions(seed_molecule=mol, environment_pdb=pdb_text)
        results_df = scorer(molecules, conditions)
    """

    def __init__(
        self,
        reward_function_names: List[str] = None,
        rescore_optimized_poses: bool = True,
    ):
        if reward_function_names is None:
            reward_function_names = [k for k in _REWARD_REGISTRY.keys()]
        self.rescore_optimized_poses = rescore_optimized_poses
        self._unidock_score_only: Optional[UniDockReward] = None
        self._unidock_local_opt: Optional[UniDockReward] = None

        if rescore_optimized_poses:
            if "unidock" not in reward_function_names:
                raise ValueError(
                    "rescore_optimized_poses requires 'unidock' in "
                    "reward_function_names."
                )
            self._unidock_score_only = UniDockReward(score_only=True)
            self._unidock_local_opt = UniDockReward(score_only=False)
            reward_function_names = [
                n for n in reward_function_names if n != "unidock"
            ]

        self.reward_functions = self._build_reward_functions(
            reward_function_names
        )

    def _build_reward_functions(
        self,
        reward_function_names: List[str],
    ) -> List[RewardFunction]:
        """Resolve all spec dicts / instances into RewardFunction objects."""

        return [get_reward_class(name)() for name in reward_function_names]

    def _parse_mol_inputs(self, input_mols: List[MoleculeType3D]) -> List[Chem.Mol]:
        mols = []
        for input_mol in input_mols:
            if isinstance(input_mol, str):
                # Keep the last tagged block. Prompts include format examples,
                # so an earlier <sdf>/<smiles_xyz>/<mol> block is not the answer.
                matches = list(re.finditer(
                    r"(?s)<sdf>(.*?)</sdf>|<sdf_enum>(.*?)</sdf_enum>|<smiles_xyz>(.*?)</smiles_xyz>|<mol>(.*?)</mol>",
                    input_mol,
                ))
                if matches:
                    input_mol = next(
                        group for group in matches[-1].groups() if group is not None
                    )

                if "\n" in input_mol:
                    mol = reconstruct_simple_sdf(input_mol)
                    if mol is None:
                        mol = reconstruct_full_sdf(input_mol)
                elif "|" in input_mol:
                    mol = reconstruct_smiles_xyz(input_mol)
                    if mol is None:
                        mol = reconstruct_smiles_enum(input_mol)
                else:  # Mal-formatted input string
                    mol = None
            # Non-string input , for example from a Diffusion Model or other Molecular
            # Generator, must already be a reconstructed RDKit Molecule object, or None
            else:  
                mol = input_mol
            if mol is None:
                logger.warning("Warning: Error parsing molecule: %s", input_mol)
            else:
                try:
                    Chem.SanitizeMol(mol)
                    Chem.AddHs(mol, addCoords=True)
                    if mol.GetNumConformers() == 0:
                        logger.warning("Warning: No conformers found in molecule: %s", input_mol)
                        mol = None
                except Exception:
                    logger.warning("Warning: Error sanitizing molecule: %s", input_mol)
                    mol = None
            mols.append(mol)
        return mols

    def __call__(
        self,
        input_mols: List[MoleculeType3D],
        conditions: Union[GenerationConditions, List[GenerationConditions]],
        parallel: bool = True,
    ) -> pd.DataFrame:
        """
        Apply all reward functions to a list of molecules.

        Args:
            mols: List of RDKit molecules.
            conditions: One :class:`~bench_3d_fit.generation_conditions.GenerationConditions`
                shared by all molecules, or a list of them (one per molecule).
            parallel: If ``True`` (default), run reward functions
                concurrently in threads.  Set to ``False`` for
                sequential execution (useful for debugging).

        Returns:
            DataFrame with per-molecule scores, pass/fail flags, and a
            weighted ``main_reward`` column.
        """
        results = pd.DataFrame()
        non_none_mask = [mol is not None for mol in input_mols]
        results["raw_input_passed"] = non_none_mask
        mols = self._parse_mol_inputs(input_mols)
        # producing a valid molecule is the technically the first condition
        # However, Mol object-outputting models will always pass it
        valid_mask = [mol is not None for mol in mols]
        results["ROMol"] = mols
        results["valid_mol_passed"] = valid_mask

        # The scoring helpers (and UniDock) skip None molecules internally and
        # return full-length, row-aligned columns, so the full mols/conditions
        # can be passed straight through and the resulting columns copied
        # across directly -- no index broadcasting needed.
        if any(valid_mask):
            if self.rescore_optimized_poses:
                valid_results = self._score_with_rescore(mols, conditions, parallel)
            else:
                valid_results = pd.DataFrame()
                self._run_rewards_on_valid(mols, conditions, valid_results,
                                           parallel=parallel)
            for col in valid_results.columns:
                results[col] = valid_results[col]
        else:
            for col in self._get_all_reward_columns():
                results[col] = [None] * len(mols)

        return results

    def _score_with_rescore(
        self,
        mols: List[Optional[Chem.Mol]],
        conditions: Union[GenerationConditions, List[GenerationConditions]],
        parallel: bool,
    ) -> pd.DataFrame:
        """Two-pass scoring: raw poses, then UniDock-optimized poses."""
        valid_results = pd.DataFrame()

        # Pass 1: raw poses — UniDock score-only first, then other rewards
        unidock_raw = self._unidock_score_only(
            copy.deepcopy(mols), conditions
        )
        self._merge_fn_results(
            self._unidock_score_only, unidock_raw, valid_results
        )
        self._run_rewards_on_valid(mols, conditions, valid_results,
                                   parallel=parallel)

        # Pass 2: optimized poses — UniDock local opt first, then other rewards
        unidock_opt = self._unidock_local_opt(
            copy.deepcopy(mols), conditions
        )
        self._merge_fn_results(
            self._unidock_local_opt,
            unidock_opt,
            valid_results,
            prefix="opt_",
        )
        # UniDock local optimization can fail for individual molecules and
        # return None for those poses. Strip Hs only from successful poses and
        # score just those, so a few failed optimizations don't void the
        # second-pass scores for the rest of the batch.
        optimized_mols = unidock_opt.get("ROMol", [None] * len(mols))
        optimized_mols = [
            Chem.RemoveHs(mol) if mol is not None else None
            for mol in optimized_mols
        ]
        self._run_rewards_on_valid(
            optimized_mols, conditions, valid_results,
            prefix="opt_", parallel=parallel,
        )
        return valid_results

    def _run_rewards_on_valid(self, mols, conditions, results, prefix: str = "",
                              parallel: bool = True):
        """Run reward functions, skipping molecules that are ``None``.

        ``None`` molecules (e.g. parse failures or failed pose optimization)
        are excluded from the reward computation and back-filled with ``None``
        in every reward column. This prevents a single ``None`` from crashing a
        reward function and voiding the scores for the whole batch.
        """
        keep_idx = [i for i, m in enumerate(mols) if m is not None]

        if not keep_idx:
            for reward_fn in self.reward_functions:
                for score_name in reward_fn.SCORE_DESCRIPTIONS.keys():
                    results[f"{prefix}{reward_fn.name}_{score_name}"] = (
                        [None] * len(mols)
                    )
            return

        subset_mols = [mols[i] for i in keep_idx]
        if isinstance(conditions, list):
            subset_conditions = [conditions[i] for i in keep_idx]
        else:
            subset_conditions = conditions

        subset_results = pd.DataFrame()
        self._run_rewards(subset_mols, subset_conditions, subset_results,
                          prefix=prefix, parallel=parallel)

        # Back-fill into full-length columns so rows align with `mols`
        # (None for any skipped molecule).
        for col in subset_results.columns:
            full_col = [None] * len(mols)
            for idx, val in zip(keep_idx, subset_results[col]):
                full_col[idx] = val
            results[col] = full_col

    def _get_all_reward_columns(self) -> List[str]:
        """Return all column names that the reward functions would produce."""
        cols = []
        for reward_fn in self.reward_functions:
            for score_name in reward_fn.SCORE_DESCRIPTIONS.keys():
                cols.append(f"{reward_fn.name}_{score_name}")
        if self.rescore_optimized_poses:
            for score_name in self._unidock_score_only.SCORE_DESCRIPTIONS.keys():
                cols.append(f"unidock_{score_name}")
            for reward_fn in self.reward_functions:
                for score_name in reward_fn.SCORE_DESCRIPTIONS.keys():
                    cols.append(f"opt_{reward_fn.name}_{score_name}")
            for score_name in self._unidock_local_opt.SCORE_DESCRIPTIONS.keys():
                cols.append(f"opt_unidock_{score_name}")
        return cols

    def _run_rewards(self, mols, conditions, results, prefix: str = "", parallel: bool = True):
        if parallel:
            self._run_rewards_parallel(mols, conditions, results, prefix=prefix)
        else:
            self._run_rewards_sequential(mols, conditions, results, prefix=prefix)

    def _run_rewards_sequential(
        self,
        mols,
        conditions,
        results,
        prefix: str = "",
    ):
        for reward_fn in self.reward_functions:
            fn_results = reward_fn(copy.deepcopy(mols), conditions)
            self._merge_fn_results(reward_fn, fn_results, results, prefix=prefix)

    def _run_rewards_parallel(
        self,
        mols,
        configuration,
        results,
        prefix: str = "",
    ):
        if not self.reward_functions:
            return
        with ThreadPoolExecutor(max_workers=len(self.reward_functions)) as pool:
            future_to_fn = {
                pool.submit(reward_fn, copy.deepcopy(mols), configuration): reward_fn
                for reward_fn in self.reward_functions
            }
            for future in as_completed(future_to_fn):
                reward_fn = future_to_fn[future]
                fn_results = future.result()
                self._merge_fn_results(reward_fn, fn_results, results, prefix=prefix)

    @staticmethod
    def _merge_fn_results(reward_fn, fn_results, results, prefix: str = ""):
        for key, values in fn_results.items():
            results[f"{prefix}{reward_fn.name}_{key}"] = values
