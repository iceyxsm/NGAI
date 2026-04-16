"""Local goodness evaluators for gradient-free training.

Inspired by Forward-Forward: each layer has a local quality
metric that doesn't require backpropagation.
"""

import torch
from torch import Tensor


class GoodnessEvaluator:
    """Evaluates layer quality using local goodness metrics.

    Args:
        threshold: Goodness threshold for positive/negative.
    """

    DEFAULT_THRESHOLD = 2.0

    def __init__(self, threshold: float = DEFAULT_THRESHOLD) -> None:
        self.threshold = threshold

    def goodness(self, activations: Tensor) -> Tensor:
        """Compute goodness (sum of squared activations).

        Args:
            activations: Tensor of shape (batch, dim).

        Returns:
            Goodness scores of shape (batch,).
        """
        return activations.pow(2).sum(dim=-1)

    def ff_loss(self, pos_acts: Tensor, neg_acts: Tensor) -> Tensor:
        """Forward-Forward style loss.

        Args:
            pos_acts: Activations from real data.
            neg_acts: Activations from corrupted data.

        Returns:
            Scalar loss.
        """
        pos_g = self.goodness(pos_acts)
        neg_g = self.goodness(neg_acts)
        pos_loss = torch.log1p(torch.exp(self.threshold - pos_g)).mean()
        neg_loss = torch.log1p(torch.exp(neg_g - self.threshold)).mean()
        return pos_loss + neg_loss
