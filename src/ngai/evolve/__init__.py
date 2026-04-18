"""NGAI Evolve: Hybrid gradient-free training system.

Combines evolutionary selection, local goodness metrics,
ternary mutations, and parallel layer training for
maximum GPU utilization without backpropagation.
"""

from ngai.evolve.cuda_batch_eval import BatchedPopulationEvaluator
from ngai.evolve.eggroll_trainer import EggrollTrainer
from ngai.evolve.es_trainer import ESTrainer
from ngai.evolve.evaluator import GoodnessEvaluator
from ngai.evolve.guided_mutator import GuidedMutator
from ngai.evolve.hybrid_trainer import HybridEvolveTrainer
from ngai.evolve.mutator import TernaryMutator
from ngai.evolve.population import Population
from ngai.evolve.trainer import EvolveTrainer

__all__ = [
    "BatchedPopulationEvaluator",
    "EggrollTrainer",
    "ESTrainer",
    "EvolveTrainer",
    "GoodnessEvaluator",
    "GuidedMutator",
    "HybridEvolveTrainer",
    "Population",
    "TernaryMutator",
]
