"""
Reward function abstract base class and name-to-class registry.

:class:`RewardFunction` is the abstract base for all reward functions.
Subclasses override :meth:`compute` with their scoring logic.  The registry
maps short string names (e.g. ``"posebusters"``) to concrete subclasses so
that :class:`~bench_3d_fit.reward.scorer.RewardScorer` can instantiate
them by name.
"""

from __future__ import annotations

import logging
import traceback
from typing import Any, Dict, List, Type, Union

from rdkit import Chem

from bench_3d_fit.generation_conditions import GenerationConditions


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Registry — maps string names to RewardFunction subclasses
# Reward Functions are registered where they are defined in rewards.py
# ---------------------------------------------------------------------------

_REWARD_REGISTRY: Dict[str, Type[RewardFunction]] = {}


def register_reward(cls: Type[RewardFunction]) -> None:
    """Register a ``RewardFunction`` subclass by its ``name`` attribute.

    Registered names can be passed to :class:`RewardScorer`
    (e.g. ``RewardScorer(["unidock", "posebusters"])``) or resolved
    individually via :func:`get_reward_class`.
    """
    _REWARD_REGISTRY[cls.name] = cls


def get_reward_class(name: str) -> Type[RewardFunction]:
    """Look up a registered ``RewardFunction`` subclass by name.

    Raises ``KeyError`` if *name* is not found.
    """
    if name not in _REWARD_REGISTRY:
        available = ", ".join(sorted(_REWARD_REGISTRY.keys()))
        raise KeyError(
            f"Unknown reward '{name}'. "
            f"Available: [{available}]. "
            f"Register with register_reward()."
        )
    return _REWARD_REGISTRY[name]


# ---------------------------------------------------------------------------
# RewardFunction base class
# ---------------------------------------------------------------------------

class RewardFunction:
    """Abstract base class for reward functions.

    Subclass and override :meth:`compute` to define a scoring criterion.
    ``compute(mols, conditions)`` receives the molecules to score and either
    one :class:`~bench_3d_fit.generation_conditions.GenerationConditions`
    (shared by all molecules) or a list of them (one per molecule).

    ``compute()`` must return a dict mapping score names to lists of values
    (one per molecule).  The keys should match :attr:`SCORE_DESCRIPTIONS`.
    """
    name: str
    SCORE_DESCRIPTIONS: Dict[str, str] = {}

    @staticmethod
    def _merge_per_mol_results(
        per_mol_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Merge single-molecule result dicts into one batch result."""
        if not per_mol_results:
            return {"score": []}
        keys = per_mol_results[0].keys()
        return {k: [r[k][0] for r in per_mol_results] for k in keys}

    def compute(
        self,
        mols: List[Chem.Mol],
        conditions: Union[GenerationConditions, List[GenerationConditions]],
    ) -> Dict[str, Any]:
        """
        Compute raw scores for a list of molecules.

        Subclasses must override this method.

        Args:
            mols: Molecules to score.
            conditions: One :class:`~bench_3d_fit.generation_conditions.GenerationConditions`
                for the whole batch, or one per molecule.

        Returns:
            A dict containing at least ``'score'`` (a list of numeric
            values, one per molecule).  Additional keys are preserved
            and forwarded to the results DataFrame.
        """
        raise NotImplementedError(
            f"Subclass {self.__class__.__name__} must implement compute() "
            f"method to return a dict with at least a 'score' key."
        )

    def __call__(
        self,
        mols: List[Chem.Mol],
        conditions: Union[GenerationConditions, List[GenerationConditions]]
    ) -> Dict[str, Any]:
        """
        Score molecules against their generation conditions.
        """
        try:
            return self.compute(mols, conditions)
        except Exception:
            logger.error(
                    f"Error computing {self.name}:\n{traceback.format_exc()}"
                )
            return {
                f"{score_name}": [None] * len(mols) \
                    for score_name in self.SCORE_DESCRIPTIONS.keys()
            }
