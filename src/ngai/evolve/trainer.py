"""The hybrid evolutionary trainer.

Combines: evolutionary selection + local goodness + ternary mutations
+ parallel evaluation. No backpropagation.
"""

import time

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from ngai.evolve.mutator import TernaryMutator
from ngai.evolve.population import Population


class EvolveTrainer:
    """Gradient-free hybrid evolutionary trainer.

    Training loop:
    1. Generate mutated variants of current weights
    2. Evaluate all variants (forward pass only)
    3. Select the best variant
    4. Replace current weights with best
    5. Repeat

    Args:
        model: The NGAI model to train.
        pop_size: Population size per step.
        mutation_rate: Fraction of weights to mutate.
        device: Device to train on.
    """

    DEFAULT_POP = 16
    DEFAULT_RATE = 0.005
    LOG_EVERY = 50

    def __init__(
        self,
        model: nn.Module,
        pop_size: int = DEFAULT_POP,
        mutation_rate: float = DEFAULT_RATE,
        device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.device = device or torch.device("cpu")
        self.mutator = TernaryMutator(mutation_rate, self.device)
        self.population = Population(pop_size)
        self.best_loss = float("inf")

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
    def _evaluate_variant(self, flat_weights: Tensor, x: Tensor, y: Tensor) -> float:
        """Evaluate one weight variant on a batch."""
        self._set_flat_weights(flat_weights)
        self.model.eval()
        logits = self.model(x)
        if isinstance(logits, tuple):
            logits = logits[0]
        vocab = logits.shape[-1]
        loss = torch.nn.functional.cross_entropy(logits.view(-1, vocab), y.view(-1))
        return loss.item()

    def train_step(self, x: Tensor, y: Tensor) -> float:
        """One evolutionary training step.

        Args:
            x: Input of shape (batch, seq_len).
            y: Target of shape (batch, seq_len).

        Returns:
            Best loss from this step.
        """
        x, y = x.to(self.device), y.to(self.device)
        base_weights = self._get_flat_weights()
        variants = self.mutator.batch_mutate(base_weights, self.population.pop_size)

        losses = torch.zeros(self.population.pop_size, device=self.device)
        for i in range(self.population.pop_size):
            losses[i] = self._evaluate_variant(variants[i], x, y)

        best_idx = losses.argmin()
        best_loss = losses[best_idx].item()

        if best_loss < self.best_loss:
            self._set_flat_weights(variants[best_idx])
            self.best_loss = best_loss

        self.population.record_fitness(best_loss)
        return best_loss

    def train(self, dataloader: DataLoader, steps: int) -> list[float]:
        """Run evolutionary training.

        Args:
            dataloader: Training data loader.
            steps: Number of training steps.

        Returns:
            List of best losses per step.
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
                avg = sum(losses[-self.LOG_EVERY:]) / self.LOG_EVERY
                elapsed = time.perf_counter() - t0
                sps = (step + 1) / elapsed
                print(
                    f"  step {step+1:>5} | loss {avg:.4f} | "
                    f"best {self.best_loss:.4f} | {sps:.1f} steps/s"
                )

        return losses
