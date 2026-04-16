"""Evolution Strategy (ES) trainer with momentum and antithetic sampling.

This implements OpenAI-style Natural Evolution Strategies adapted for
ternary weights. Key improvements over naive evolutionary search:

1. Antithetic sampling: for each noise vector, evaluate both +noise
   and -noise. This halves variance for free.
2. Fitness shaping: rank-based fitness normalization prevents outliers
   from dominating the update.
3. Momentum: exponential moving average of the update direction,
   biasing future mutations toward historically good directions.
4. Adaptive noise: scale noise based on recent improvement rate.

Reference: Salimans et al. "Evolution Strategies as a Scalable
Alternative to Reinforcement Learning" (2017), adapted for
discrete ternary weight spaces.
"""

import time

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader


class ESTrainer:
    """Natural Evolution Strategy trainer for ternary models.

    Instead of random mutation + selection, ES computes a weighted
    average of perturbation directions based on their fitness.
    This gives a gradient-like signal without backpropagation.

    Args:
        model: The NGAI model to train.
        pop_size: Number of perturbation pairs (total evals = 2x this).
        sigma: Noise standard deviation.
        lr: Learning rate for weight updates.
        momentum: Momentum coefficient for update direction.
        device: Device to train on.
    """

    DEFAULT_POP = 16
    DEFAULT_SIGMA = 0.1
    DEFAULT_LR = 0.01
    DEFAULT_MOMENTUM = 0.9
    LOG_EVERY = 50
    SIGMA_DECAY = 0.999
    MIN_SIGMA = 0.01

    def __init__(
        self,
        model: nn.Module,
        pop_size: int = DEFAULT_POP,
        sigma: float = DEFAULT_SIGMA,
        lr: float = DEFAULT_LR,
        momentum: float = DEFAULT_MOMENTUM,
        device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.device = device or torch.device("cpu")
        self.pop_size = pop_size
        self.sigma = sigma
        self.lr = lr
        self.momentum = momentum
        self.best_loss = float("inf")
        self.step_count = 0

        self._total_params = sum(p.numel() for p in model.parameters())
        self._velocity = torch.zeros(self._total_params, device=self.device)

    def _get_flat_weights(self) -> Tensor:
        """Flatten all model weights into a single tensor."""
        return torch.cat([p.data.view(-1) for p in self.model.parameters()])

    def _set_flat_weights(self, flat: Tensor) -> None:
        """Set model weights from a flat tensor."""
        offset = 0
        for p in self.model.parameters():
            numel = p.numel()
            p.data.copy_(flat[offset : offset + numel].view(p.shape))
            offset += numel

    @torch.no_grad()
    def _evaluate(self, flat_weights: Tensor, x: Tensor, y: Tensor) -> float:
        """Evaluate one weight configuration."""
        self._set_flat_weights(flat_weights)
        self.model.eval()
        logits = self.model(x)
        if isinstance(logits, tuple):
            logits = logits[0]
        vocab = logits.shape[-1]
        return torch.nn.functional.cross_entropy(
            logits.view(-1, vocab), y.view(-1),
        ).item()

    @staticmethod
    def _fitness_shaping(losses: Tensor) -> Tensor:
        """Rank-based fitness shaping.

        Converts raw losses to rank-based utilities in [-0.5, 0.5].
        This makes the update robust to outliers and loss scale.

        Args:
            losses: Raw loss values of shape (n,).

        Returns:
            Shaped fitness of shape (n,), centered around 0.
        """
        n = losses.shape[0]
        ranks = torch.zeros_like(losses)
        sorted_indices = losses.argsort()
        for rank, idx in enumerate(sorted_indices):
            ranks[idx] = rank
        utilities = ranks / (n - 1) - 0.5
        return -utilities

    def _requantize_ternary(self, weights: Tensor) -> Tensor:
        """Snap continuous weights back to ternary {-1, 0, +1}.

        Uses threshold-based rounding: values above mean absolute
        value get sign-preserved, below get zeroed.

        Args:
            weights: Continuous weight tensor.

        Returns:
            Ternary weight tensor.
        """
        abs_w = weights.abs()
        threshold = abs_w.mean()
        sign = weights.sign()
        return torch.where(abs_w > threshold, sign, torch.zeros_like(weights))

    @torch.no_grad()
    def train_step(self, x: Tensor, y: Tensor) -> float:
        """One ES training step with antithetic sampling.

        For each of pop_size noise vectors:
        1. Evaluate weights + sigma*noise (positive perturbation)
        2. Evaluate weights - sigma*noise (antithetic perturbation)
        3. Compute fitness-shaped update from all 2*pop_size evals
        4. Apply momentum-accelerated update
        5. Re-quantize to ternary

        Args:
            x: Input of shape (batch, seq_len).
            y: Target of shape (batch, seq_len).

        Returns:
            Loss of the current (unperturbed) weights.
        """
        x, y = x.to(self.device), y.to(self.device)
        base_weights = self._get_flat_weights()

        base_loss = self._evaluate(base_weights, x, y)

        noise = torch.randn(
            self.pop_size, self._total_params, device=self.device,
        )

        all_losses = torch.zeros(self.pop_size * 2, device=self.device)
        for i in range(self.pop_size):
            pos_weights = base_weights + self.sigma * noise[i]
            neg_weights = base_weights - self.sigma * noise[i]
            all_losses[i * 2] = self._evaluate(pos_weights, x, y)
            all_losses[i * 2 + 1] = self._evaluate(neg_weights, x, y)

        shaped = self._fitness_shaping(all_losses)

        grad_estimate = torch.zeros(self._total_params, device=self.device)
        for i in range(self.pop_size):
            diff = shaped[i * 2] - shaped[i * 2 + 1]
            grad_estimate += diff * noise[i]
        grad_estimate /= (self.pop_size * self.sigma)

        self._velocity = (
            self.momentum * self._velocity + (1 - self.momentum) * grad_estimate
        )

        new_weights = base_weights + self.lr * self._velocity
        new_weights = self._requantize_ternary(new_weights)

        new_loss = self._evaluate(new_weights, x, y)
        if new_loss < base_loss:
            self._set_flat_weights(new_weights)
            step_loss = new_loss
        else:
            self._set_flat_weights(base_weights)
            step_loss = base_loss

        self.best_loss = min(self.best_loss, step_loss)
        self.sigma = max(self.MIN_SIGMA, self.sigma * self.SIGMA_DECAY)
        self.step_count += 1
        return step_loss

    def train(self, dataloader: DataLoader, steps: int) -> list[float]:
        """Run ES training loop.

        Args:
            dataloader: Training data loader.
            steps: Number of training steps.

        Returns:
            List of losses per step.
        """
        losses: list[float] = []
        loader_iter = iter(dataloader)
        t0 = time.perf_counter()

        for step in range(steps):
            try:
                x, y = next(loader_iter)
            except StopIteration:
                loader_iter = iter(dataloader)
                x, y = next(loader_iter)

            loss = self.train_step(x, y)
            losses.append(loss)

            if (step + 1) % self.LOG_EVERY == 0:
                avg = sum(losses[-self.LOG_EVERY :]) / self.LOG_EVERY
                elapsed = time.perf_counter() - t0
                sps = (step + 1) / elapsed
                print(
                    f"  step {step + 1:>5} | loss {avg:.4f} | "
                    f"best {self.best_loss:.4f} | {sps:.1f} steps/s | "
                    f"sigma {self.sigma:.4f}"
                )

        return losses
