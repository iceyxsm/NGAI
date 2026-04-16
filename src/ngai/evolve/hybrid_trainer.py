"""Hybrid evolutionary trainer with CUDA batching and guided mutation.

The full pipeline:
1. Forward pass to collect per-layer activations
2. Compute goodness scores -> update per-layer mutation rates
3. Generate variants with guided mutation (weak layers mutated more)
4. Evaluate all variants via CUDA-batched forward passes
5. Select best variant (elitist selection)
6. Repeat

This is the core of the "no backprop" approach. Every component
is designed to maximize GPU utilization while searching the weight
space intelligently via local quality signals.
"""

import time

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from ngai.evolve.cuda_batch_eval import BatchedPopulationEvaluator
from ngai.evolve.guided_mutator import GuidedMutator
from ngai.evolve.population import Population


class HybridEvolveTrainer:
    """Gradient-free hybrid evolutionary trainer with guided mutation.

    Combines:
    - CUDA-batched population evaluation (GPU saturation)
    - Forward-Forward goodness-guided per-layer mutation
    - Elitist selection with population tracking

    Args:
        model: The NGAI model to train.
        pop_size: Population size per step.
        mutation_rate: Base mutation rate.
        goodness_scale: How strongly goodness guides mutation.
        goodness_interval: Steps between goodness updates.
        device: Device to train on.
    """

    DEFAULT_POP = 64
    DEFAULT_RATE = 0.001
    DEFAULT_GOODNESS_SCALE = 2.0
    GOODNESS_INTERVAL = 10
    WARMUP_STEPS = 50
    LOG_EVERY = 50

    def __init__(
        self,
        model: nn.Module,
        pop_size: int = DEFAULT_POP,
        mutation_rate: float = DEFAULT_RATE,
        goodness_scale: float = DEFAULT_GOODNESS_SCALE,
        goodness_interval: int = GOODNESS_INTERVAL,
        warmup_steps: int = WARMUP_STEPS,
        device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.device = device or torch.device("cpu")
        self.pop_size = pop_size
        self.goodness_interval = goodness_interval
        self.warmup_steps = warmup_steps

        self.guided_mutator = GuidedMutator(
            model, base_rate=mutation_rate,
            goodness_scale=goodness_scale, device=self.device,
        )
        self.evaluator = BatchedPopulationEvaluator(model, self.device)
        self.population = Population(pop_size)
        self.best_loss = float("inf")
        self.step_count = 0

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
    def train_step(self, x: Tensor, y: Tensor) -> float:
        """One hybrid evolutionary training step.

        Pipeline:
        1. Optionally update goodness-guided mutation rates
        2. Generate guided variants
        3. Evaluate all variants + current weights
        4. Select best, update model

        Args:
            x: Input of shape (batch, seq_len).
            y: Target of shape (batch, seq_len).

        Returns:
            Best loss from this step.
        """
        x, y = x.to(self.device), y.to(self.device)
        base_weights = self._get_flat_weights()

        use_goodness = (
            self.step_count >= self.warmup_steps
            and self.step_count % self.goodness_interval == 0
        )
        if use_goodness:
            loss, activations = self.evaluator.evaluate_with_goodness(
                base_weights, x, y,
            )
            self.guided_mutator.update_rates_from_goodness(activations)
        else:
            self.evaluator._inject_weights(base_weights)
            self.model.eval()
            logits = self.model(x)
            if isinstance(logits, tuple):
                logits = logits[0]
            vocab = logits.shape[-1]
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, vocab), y.view(-1),
            ).item()

        variants = self.guided_mutator.guided_mutate(
            base_weights, self.pop_size,
        )
        variant_losses = self.evaluator.evaluate_population(variants, x, y)

        best_idx = variant_losses.argmin()
        best_variant_loss = variant_losses[best_idx].item()

        if best_variant_loss < loss:
            self._set_flat_weights(variants[best_idx])
            step_best = best_variant_loss
        else:
            self._set_flat_weights(base_weights)
            step_best = loss

        self.best_loss = min(self.best_loss, step_best)
        self.population.record_fitness(step_best)
        self.step_count += 1
        return step_best

    def train(self, dataloader: DataLoader, steps: int) -> list[float]:
        """Run hybrid evolutionary training.

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
                avg = sum(losses[-self.LOG_EVERY :]) / self.LOG_EVERY
                elapsed = time.perf_counter() - t0
                sps = (step + 1) / elapsed
                rates = self.guided_mutator._layer_rates
                rate_str = " ".join(f"{r:.5f}" for r in rates[:4])
                print(
                    f"  step {step + 1:>5} | loss {avg:.4f} | "
                    f"best {self.best_loss:.4f} | {sps:.1f} steps/s | "
                    f"rates [{rate_str}...]"
                )

        return losses

    def get_layer_diagnostics(self) -> dict[str, float]:
        """Get current per-layer mutation rates for monitoring.

        Returns:
            Dict mapping layer name to current mutation rate.
        """
        return {
            layer.name: rate
            for layer, rate in zip(
                self.guided_mutator.layers,
                self.guided_mutator._layer_rates,
                strict=True,
            )
        }
