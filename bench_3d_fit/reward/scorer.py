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
from typing import Any, Dict, List, Union

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
    ):
        if reward_function_names is None:
            reward_function_names = [k for k in _REWARD_REGISTRY.keys()]
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
                match = re.search(
                    r"(?:<sdf>(.*?)</sdf>|<smiles_xyz>(.*?)</smiles_xyz>)\s*$",
                    input_mol,
                    re.DOTALL
                )
                if match:
                    input_mol = match.group(1) if match.group(1) is not None else match.group(2)

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
        mols = self._parse_mol_inputs(input_mols)
        # producing a valid molecule is the technically the first condition
        # However, Mol object-outputting models will always pass it
        valid_mask = [mol is not None for mol in mols]
        results["ROMol"] = mols
        results["valid_mol_passed"] = valid_mask

        valid_mols = [m for m, v in zip(mols, valid_mask) if v]
        if isinstance(conditions, list):
            valid_conditions = [c for c, v in zip(conditions, valid_mask) if v]
        else:
            valid_conditions = conditions

        if valid_mols:
            valid_results = pd.DataFrame()
            if parallel:
                self._run_rewards_parallel(valid_mols, valid_conditions, valid_results)
            else:
                self._run_rewards_sequential(valid_mols, valid_conditions, valid_results)

            valid_idx = [i for i, v in enumerate(valid_mask) if v]
            for col in valid_results.columns:
                full_col = [None] * len(mols)
                for idx, val in zip(valid_idx, valid_results[col]):
                    full_col[idx] = val
                results[col] = full_col
        else:
            for col in self._get_all_reward_columns():
                results[col] = [None] * len(mols)

        return results

    def _get_all_reward_columns(self) -> List[str]:
        """Return all column names that the reward functions would produce."""
        cols = []
        for reward_fn in self.reward_functions:
            for score_name in reward_fn.SCORE_DESCRIPTIONS.keys():
                cols.append(f"{reward_fn.name}_{score_name}")
        return cols

    def _run_rewards_sequential(self, mols, conditions, results):
        for reward_fn in self.reward_functions:
            fn_results = reward_fn(copy.deepcopy(mols), conditions)
            self._merge_fn_results(reward_fn, fn_results, results)

    def _run_rewards_parallel(self, mols, configuration, results):
        with ThreadPoolExecutor(max_workers=len(self.reward_functions)) as pool:
            future_to_fn = {
                pool.submit(reward_fn, copy.deepcopy(mols), configuration): reward_fn
                for reward_fn in self.reward_functions
            }
            for future in as_completed(future_to_fn):
                reward_fn = future_to_fn[future]
                fn_results = future.result()
                self._merge_fn_results(reward_fn, fn_results, results)

    @staticmethod
    def _merge_fn_results(reward_fn, fn_results, results):
        for key, values in fn_results.items():
            results[f"{reward_fn.name}_{key}"] = values
