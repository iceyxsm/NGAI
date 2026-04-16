"""Population management for evolutionary training.

Maintains a population of model weight variants,
handles selection, and tracks fitness history.
"""

import torch
from torch import Tensor


class Population:
    """Manages a population of ternary weight variants.

    Args:
        pop_size: Number of individuals in the population.
        elite_fraction: Fraction to keep as elite.
    """

    DEFAULT_POP = 16
    DEFAULT_ELITE = 0.25

    def __init__(
        self,
        pop_size: int = DEFAULT_POP,
        elite_fraction: float = DEFAULT_ELITE,
    ) -> None:
        self.pop_size = pop_size
        self.elite_count = max(1, int(pop_size * elite_fraction))
        self.fitness_history: list[float] = []

    def select(self, losses: Tensor) -> Tensor:
        """Select elite indices (lower loss = better).

        Args:
            losses: Loss values of shape (pop_size,).

        Returns:
            Indices of elite individuals.
        """
        _, indices = torch.topk(losses, self.elite_count, largest=False)
        return indices

    def record_fitness(self, best_loss: float) -> None:
        """Record the best fitness for history tracking."""
        self.fitness_history.append(best_loss)
