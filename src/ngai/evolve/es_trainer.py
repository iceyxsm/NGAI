"""Evolution Strategy (ES) trainer with momentum and antithetic sampling.

This implements OpenAI-style Natural Evolution Strategies adapted for
ternary weights. Key improvements over naive evolutionary search:

1. Antithetic sampling: for each noise vector, evaluate both +noise
   and -noise. This halves variance for free.
2. Fitness shaping: rank-based fitness normalization prevents outliers
   from dominating the update.
3. Momentum: exponential moving average of the update direction,
   biasing future mutations toward historically good directions.
4. Cosine LR schedule: high LR for exploration, decays to fine-tune.
5. Multi-batch evaluation: average loss over multiple batches per
   variant to reduce noise in the gradient estimate.

Reference: Salimans et al. "Evolution Strategies as a Scalable
Alternative to Reinforcement Learning" (2017), adapted for
discrete ternary weight spaces.
"""

import math
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
        lr: Peak learning rate for weight updates.
        lr_min_ratio: Minimum LR as fraction of peak (for cosine schedule).
        momentum: Momentum coefficient for update direction.
        weight_decay: L2 weight decay coefficient.
        eval_batches: Number of batches to average per variant evaluation.
        device: Device to train on.
    """

    DEFAULT_POP = 16
    DEFAULT_SIGMA = 0.1
    DEFAULT_LR = 0.01
    DEFAULT_LR_MIN_RATIO = 0.1
    DEFAULT_MOMENTUM = 0.9
    DEFAULT_WEIGHT_DECAY = 0.001
    DEFAULT_EVAL_BATCHES = 1
    LOG_EVERY = 50

    def __init__(
        self,
        model: nn.Module,
        pop_size: int = DEFAULT_POP,
        sigma: float = DEFAULT_SIGMA,
        lr: float = DEFAULT_LR,
        lr_min_ratio: float = DEFAULT_LR_MIN_RATIO,
        momentum: float = DEFAULT_MOMENTUM,
        weight_decay: float = DEFAULT_WEIGHT_DECAY,
        eval_batches: int = DEFAULT_EVAL_BATCHES,
        device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.device = device or torch.device("cpu")
        self.pop_size = pop_size
        self.sigma = sigma
        self.lr_peak = lr
        self.lr_min = lr * lr_min_ratio
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.eval_batches = eval_batches
        self.best_loss = float("inf")
        self.step_count = 0
        self.total_steps = 0

        self._total_params = sum(p.numel() for p in model.parameters())
        self._velocity = torch.zeros(self._total_params, device=self.device)

        self._graph_eval = None
        if self.device.type == "cuda":
            from ngai.evolve.cuda_graph_eval import CUDAGraphEvaluator
            self._graph_eval = CUDAGraphEvaluator(model, self.device)
        self._graph_captured = False

    def _cosine_lr(self) -> float:
        """Compute current LR using cosine annealing schedule.

        Returns:
            Current learning rate.
        """
        if self.total_steps <= 0:
            return self.lr_peak
        progress = min(1.0, self.step_count / self.total_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.lr_min + (self.lr_peak - self.lr_min) * cosine

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
        """Evaluate one weight configuration on a single batch."""
        self._set_flat_weights(flat_weights)
        self.model.eval()
        logits = self.model(x)
        if isinstance(logits, tuple):
            logits = logits[0]
        vocab = logits.shape[-1]
        return torch.nn.functional.cross_entropy(
            logits.view(-1, vocab), y.view(-1),
        ).item()

    @torch.no_grad()
    def _evaluate_multi(
        self, flat_weights: Tensor, batches: list[tuple[Tensor, Tensor]],
    ) -> float:
        """Evaluate weights averaged over multiple batches.

        Args:
            flat_weights: Flat weight tensor.
            batches: List of (x, y) batch tuples.

        Returns:
            Average loss across all batches.
        """
        total = 0.0
        for x, y in batches:
            total += self._evaluate(flat_weights, x, y)
        return total / len(batches)

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

    def _get_eval_batches(
        self, loader_iter: object, dataloader: DataLoader,
    ) -> tuple[list[tuple[Tensor, Tensor]], object]:
        """Fetch eval_batches worth of data from the loader.

        Args:
            loader_iter: Current dataloader iterator.
            dataloader: The dataloader to reset from.

        Returns:
            (batches, updated_loader_iter)
        """
        batches: list[tuple[Tensor, Tensor]] = []
        for _ in range(self.eval_batches):
            try:
                x, y = next(loader_iter)
            except StopIteration:
                loader_iter = iter(dataloader)
                x, y = next(loader_iter)
            batches.append((x.to(self.device), y.to(self.device)))
        return batches, loader_iter

    @torch.no_grad()
    def _evaluate_all_variants(
        self,
        all_flat_weights: Tensor,
        batches: list[tuple[Tensor, Tensor]],
    ) -> Tensor:
        """Evaluate all variants, using CUDA graphs if available.

        Args:
            all_flat_weights: Shape (n_variants, total_params).
            batches: List of (x, y) batch tuples.

        Returns:
            Losses of shape (n_variants,).
        """
        x, y = batches[0]

        if self._graph_eval is not None and len(batches) == 1:
            if not self._graph_captured:
                try:
                    self._graph_eval.capture(x, y)
                    self._graph_captured = True
                except RuntimeError:
                    self._graph_eval = None

            if self._graph_captured:
                return self._graph_eval.evaluate_population(
                    all_flat_weights, x, y,
                )

        n_variants = all_flat_weights.shape[0]
        losses = torch.zeros(n_variants, device=self.device)
        for i in range(n_variants):
            losses[i] = self._evaluate_multi(all_flat_weights[i], batches)
        return losses

    @torch.no_grad()
    def train_step(self, batches: list[tuple[Tensor, Tensor]]) -> float:
        """One ES training step with antithetic sampling.

        For each of pop_size noise vectors:
        1. Evaluate weights + sigma*noise (positive perturbation)
        2. Evaluate weights - sigma*noise (antithetic perturbation)
        3. Compute fitness-shaped update from all 2*pop_size evals
        4. Apply momentum-accelerated update with cosine LR

        Args:
            batches: List of (x, y) batch tuples for evaluation.

        Returns:
            Loss of the updated weights.
        """
        base_weights = self._get_flat_weights()

        noise = torch.randn(
            self.pop_size, self._total_params, device=self.device,
        )

        # Build all perturbed weight vectors: [pos_0, neg_0, pos_1, neg_1, ...]
        pos_weights = base_weights.unsqueeze(0) + self.sigma * noise
        neg_weights = base_weights.unsqueeze(0) - self.sigma * noise
        # Interleave: (2*pop_size, total_params)
        all_weights = torch.stack(
            [pos_weights, neg_weights], dim=1,
        ).view(-1, self._total_params)

        all_losses = self._evaluate_all_variants(all_weights, batches)

        shaped = self._fitness_shaping(all_losses)

        # Vectorized gradient estimate: sum of (shaped_pos - shaped_neg) * noise
        pos_shaped = shaped[0::2]  # even indices
        neg_shaped = shaped[1::2]  # odd indices
        diffs = (pos_shaped - neg_shaped).unsqueeze(1)  # (pop_size, 1)
        grad_estimate = (diffs * noise).sum(dim=0) / (self.pop_size * self.sigma)

        self._velocity = (
            self.momentum * self._velocity + (1 - self.momentum) * grad_estimate
        )

        current_lr = self._cosine_lr()
        new_weights = (
            base_weights + current_lr * self._velocity
            - current_lr * self.weight_decay * base_weights
        )

        self._set_flat_weights(new_weights)
        step_loss = self._evaluate_multi(new_weights, batches)

        self.best_loss = min(self.best_loss, step_loss)
        self.step_count += 1
        return step_loss

    def train(self, dataloader: DataLoader, steps: int) -> list[float]:
        """Run ES training loop with cosine LR and multi-batch eval.

        Args:
            dataloader: Training data loader.
            steps: Number of training steps.

        Returns:
            List of losses per step.
        """
        self.total_steps = steps
        losses: list[float] = []
        loader_iter = iter(dataloader)
        t0 = time.perf_counter()

        for step in range(steps):
            batches, loader_iter = self._get_eval_batches(
                loader_iter, dataloader,
            )

            loss = self.train_step(batches)
            losses.append(loss)

            if (step + 1) % self.LOG_EVERY == 0:
                avg = sum(losses[-self.LOG_EVERY :]) / self.LOG_EVERY
                elapsed = time.perf_counter() - t0
                sps = (step + 1) / elapsed
                current_lr = self._cosine_lr()
                print(
                    f"  step {step + 1:>5} | loss {avg:.4f} | "
                    f"best {self.best_loss:.4f} | {sps:.1f} steps/s | "
                    f"lr {current_lr:.5f}"
                )

        return losses
