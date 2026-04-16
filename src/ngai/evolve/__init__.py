"""NGAI Evolve: Hybrid gradient-free training system.

Combines evolutionary selection, local goodness metrics,
ternary mutations, and parallel layer training for
maximum GPU utilization without backpropagation.
"""

from ngai.evolve.evaluator import GoodnessEvaluator
from ngai.evolve.mutator import TernaryMutator
from ngai.evolve.population import Population
from ngai.evolve.trainer import EvolveTrainer

__all__ = ["EvolveTrainer", "GoodnessEvaluator", "Population", "TernaryMutator"]
