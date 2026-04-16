"""Ternary weight mutation strategies.

Mutations operate directly on ternary weights {-1, 0, +1}
by flipping random values. No gradients needed.
"""

import torch
from torch import Tensor


class TernaryMutator:
    """Mutates ternary weights by flipping random values.

    Args:
        mutation_rate: Fraction of weights to flip per mutation.
        device: Device to create tensors on.
    """

    DEFAULT_RATE = 0.01

    def __init__(
        self,
        mutation_rate: float = DEFAULT_RATE,
        device: torch.device | None = None,
    ) -> None:
        self.mutation_rate = mutation_rate
        self.device = device or torch.device("cpu")

    def mutate(self, weights: Tensor) -> Tensor:
        """Create a mutated copy of ternary weights.

        Args:
            weights: Ternary weight tensor.

        Returns:
            New tensor with mutated weights.
        """
        mask = torch.rand_like(weights) < self.mutation_rate
        random_vals = torch.randint(
            -1, 2, weights.shape, device=weights.device
        ).float()
        return torch.where(mask, random_vals, weights)

    def batch_mutate(self, weights: Tensor, n_variants: int) -> Tensor:
        """Create multiple mutated variants.

        Args:
            weights: Base weights of shape (*shape).
            n_variants: Number of variants to create.

        Returns:
            Tensor of shape (n_variants, *shape).
        """
        expanded = weights.unsqueeze(0).expand(n_variants, *weights.shape)
        masks = (
            torch.rand(n_variants, *weights.shape, device=weights.device)
            < self.mutation_rate
        )
        random_vals = torch.randint(
            -1, 2, (n_variants, *weights.shape), device=weights.device
        ).float()
        return torch.where(masks, random_vals, expanded)
