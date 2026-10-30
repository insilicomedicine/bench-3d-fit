"""
3D-Fit Benchmark -- scoring and evaluation framework for 3D generative
chemistry models.  Generation models live outside this package; you supply
molecules (as RDKit ``Mol`` objects or formatted strings) and receive scored
DataFrames and aggregated metrics for evaluation.
"""

__version__ = "0.2.0"

try:
    from prody import confProDy
    confProDy(verbosity="none")
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
except Exception:
    pass

from bench_3d_fit.benchmark import (
    get_generation_conditions,
    write_generation_results,
    get_success_stats,
    write_success_stats,
)
from bench_3d_fit.generation_conditions import GenerationConditions
from bench_3d_fit.sampling import sample_generation_conditions
from bench_3d_fit.reward import (
    RewardScorer,
    RewardFunction,
    register_reward,
    get_reward_class,
)

__all__ = [
    "get_generation_conditions",
    "write_generation_results",
    "get_success_stats",
    "write_success_stats",
    "GenerationConditions",
    "sample_generation_conditions",
    "RewardScorer",
    "RewardFunction",
    "register_reward",
    "get_reward_class",
]
