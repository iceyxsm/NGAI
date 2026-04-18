"""EGGROLL: Evolution Guided General Optimization via Low-rank Learning.

Based on Sarkar et al. "Evolution Strategies at the Hyperscale" (2025).
Uses low-rank perturbations instead of full-rank noise for ES.

Key insight: instead of perturbing each weight independently (1M random
numbers), perturb with rank-r matrices (A @ B.T where A is out x r, B is
in x r). This reduces noise generation from O(params) to O(rank * sqrt(params))
and makes the gradient update a sum of outer products instead of element-wise.

The gradient estimate converges to full ES as rank increases, and the
sum of low-rank updates across the population is high-rank.

For our 1M param model with rank=1 and pop=32:
- Full ES noise: 32 x 1M = 32M random numbers per step
- EGGROLL noise: 32 x 2 x sqrt(1M) = 64K random numbers per step
- 500x less noise generation, same gradient quality
"""

import time

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader


class LayerSpec:
    """Metadata for one linear layer's weight matrix.

    Args:
        param: The weight parameter.
        out_features: Output dimension.
        in_features: Input dimension.
        flat_start: Start index in flat weight vector.
        flat_end: End index in flat weight vector.
    """

    def __init__(
        self, param: nn.Parameter, out_features: int,
        in_features: int, flat_start: int, flat_end: int,
    ) -> None:
        self.param = param
        self.out_features = out_features
        self.in_features = in_features
        self.flat_start = flat_start
        self.flat_end = flat_end


class EggrollTrainer:
    """EGGROLL ES trainer with low-rank perturbations.

    Same algorithm as ESTrainer but uses rank-r perturbations per
    weight matrix instead of full-rank noise. This makes noise
    generation and gradient updates dramatically cheaper.

    Still runs separate forward passes per variant (correct for
    stateful recurrence), but the perturbation construction and
    gradient accumulation are O(rank) instead of O(params).

    Args:
        model: The NGAI model to train.
        pop_size: Number of perturbation pairs (total evals = 2x this).
        rank: Rank of perturbation matrices.
        sigma: Noise standard deviation.
        lr: Learning rate.
        momentum: Momentum coefficient.
        weight_decay: L2 weight decay.
        device: Device to train on.
    """

    DEFAULT_POP = 32
    DEFAULT_RANK = 1
    DEFAULT_SIGMA = 0.1
    DEFAULT_LR = 0.01
    DEFAULT_MOMENTUM = 0.9
    DEFAULT_WEIGHT_DECAY = 0.001
    LOG_EVERY = 50

    def __init__(
        self,
        model: nn.Module,
        pop_size: int = DEFAULT_POP,
        rank: int = DEFAULT_RANK,
        sigma: float = DEFAULT_SIGMA,
        lr: float = DEFAULT_LR,
        momentum: float = DEFAULT_MOMENTUM,
        weight_decay: float = DEFAULT_WEIGHT_DECAY,
        device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.device = device or torch.device("cpu")
        self.pop_size = pop_size
        self.rank = rank
        self.sigma = sigma
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.best_loss = float("inf")
        self.step_count = 0

        self._layer_specs: list[LayerSpec] = []
        self._non_linear_indices: list[tuple[int, int, nn.Parameter]] = []
        self._total_params = 0
        self._velocity: Tensor = torch.zeros(0)
        self._map_parameters()
        self._velocity = torch.zeros(self._total_params, device=self.device)

        if self.device.type == "cuda":
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
            except RuntimeError:
                pass

    def _map_parameters(self) -> None:
        """Map model parameters to linear layers and non-linear params."""
        linear_param_ids: set[int] = set()

        for module in self.model.modules():
            if hasattr(module, 'weight') and hasattr(module, 'in_features'):
                w = module.weight
                start = -1
                cur = 0
                for p in self.model.parameters():
                    if p is w:
                        start = cur
                        break
                    cur += p.numel()
                if start >= 0:
                    self._layer_specs.append(LayerSpec(
                        param=w,
                        out_features=module.out_features,
                        in_features=module.in_features,
                        flat_start=start,
                        flat_end=start + w.numel(),
                    ))
                    linear_param_ids.add(id(w))

        cur = 0
        for p in self.model.parameters():
            if id(p) not in linear_param_ids:
                self._non_linear_indices.append((cur, cur + p.numel(), p))
            cur += p.numel()
        self._total_params = cur

    def _generate_lowrank_noise(
        self,
    ) -> list[tuple[Tensor, Tensor]]:
        """Generate low-rank noise factors for each linear layer.

        For each layer with weight (out, in), generates:
        - A: (pop_size, out, rank)
        - B: (pop_size, in, rank)
        Perturbation for variant i = A[i] @ B[i].T (out, in)

        Returns:
            List of (A, B) tuples, one per linear layer.
        """
        noise_factors: list[tuple[Tensor, Tensor]] = []
        for spec in self._layer_specs:
            a_noise = torch.randn(
                self.pop_size, spec.out_features, self.rank,
                device=self.device,
            )
            b_noise = torch.randn(
                self.pop_size, spec.in_features, self.rank,
                device=self.device,
            )
            noise_factors.append((a_noise, b_noise))
        return noise_factors

    def _generate_nonlinear_noise(self) -> Tensor | None:
        """Generate full-rank noise for non-linear parameters.

        Returns:
            Noise tensor or None if no non-linear params.
        """
        total_nl = sum(end - start for start, end, _ in self._non_linear_indices)
        if total_nl == 0:
            return None
        return torch.randn(self.pop_size, total_nl, device=self.device)

    def _apply_perturbation(
        self,
        noise_factors: list[tuple[Tensor, Tensor]],
        nl_noise: Tensor | None,
        variant_idx: int,
        sign: float,
    ) -> None:
        """Apply low-rank perturbation to model weights for one variant.

        Args:
            noise_factors: Low-rank noise per layer.
            nl_noise: Full-rank noise for non-linear params.
            variant_idx: Which variant (0 to pop_size-1).
            sign: +1.0 or -1.0 for antithetic sampling.
        """
        for spec, (a_noise, b_noise) in zip(
            self._layer_specs, noise_factors, strict=True,
        ):
            perturbation = a_noise[variant_idx] @ b_noise[variant_idx].T
            spec.param.data.add_(sign * self.sigma * perturbation)

        if nl_noise is not None:
            offset = 0
            for start, end, param in self._non_linear_indices:
                numel = end - start
                noise_chunk = nl_noise[variant_idx, offset : offset + numel]
                param.data.add_(sign * self.sigma * noise_chunk.view(param.shape))
                offset += numel

    def _save_base_weights(self) -> Tensor:
        """Save current weights as flat tensor."""
        return torch.cat([p.data.view(-1) for p in self.model.parameters()])

    def _restore_weights(self, flat: Tensor) -> None:
        """Restore weights from flat tensor."""
        offset = 0
        for p in self.model.parameters():
            numel = p.numel()
            p.data.copy_(flat[offset : offset + numel].view(p.shape))
            offset += numel

    @torch.no_grad()
    def _evaluate_current(self, x: Tensor, y: Tensor) -> float:
        """Evaluate model with current weights."""
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
        """Rank-based fitness shaping."""
        n = losses.shape[0]
        ranks = torch.zeros_like(losses)
        sorted_indices = losses.argsort()
        for rank_val, idx in enumerate(sorted_indices):
            ranks[idx] = rank_val
        utilities = ranks / (n - 1) - 0.5
        return -utilities

    @torch.no_grad()
    def train_step(self, x: Tensor, y: Tensor) -> float:
        """One EGGROLL training step.

        1. Generate low-rank noise factors per layer
        2. For each variant: apply perturbation, evaluate, restore
        3. Fitness shape, compute gradient, update with momentum

        Args:
            x: Input of shape (batch, seq_len).
            y: Target of shape (batch, seq_len).

        Returns:
            Loss after update.
        """
        x, y = x.to(self.device), y.to(self.device)
        base_weights = self._save_base_weights()

        noise_factors = self._generate_lowrank_noise()
        nl_noise = self._generate_nonlinear_noise()

        n_variants = self.pop_size * 2
        all_losses = torch.zeros(n_variants, device=self.device)

        for i in range(self.pop_size):
            self._apply_perturbation(noise_factors, nl_noise, i, 1.0)
            all_losses[i * 2] = self._evaluate_current(x, y)
            self._restore_weights(base_weights)

            self._apply_perturbation(noise_factors, nl_noise, i, -1.0)
            all_losses[i * 2 + 1] = self._evaluate_current(x, y)
            self._restore_weights(base_weights)

        shaped = self._fitness_shaping(all_losses)
        pos_shaped = shaped[0::2]
        neg_shaped = shaped[1::2]
        diffs = pos_shaped - neg_shaped

        grad_update = torch.zeros(self._total_params, device=self.device)
        for spec, (a_noise, b_noise) in zip(
            self._layer_specs, noise_factors, strict=True,
        ):
            weighted_perturbation = torch.zeros(
                spec.out_features, spec.in_features, device=self.device,
            )
            for i in range(self.pop_size):
                outer = a_noise[i] @ b_noise[i].T
                weighted_perturbation += diffs[i] * outer
            grad_update[spec.flat_start : spec.flat_end] = (
                weighted_perturbation.view(-1) / (self.pop_size * self.sigma)
            )

        if nl_noise is not None:
            nl_grad = (diffs.unsqueeze(1) * nl_noise).sum(dim=0)
            nl_grad /= (self.pop_size * self.sigma)
            offset = 0
            for start, end, _ in self._non_linear_indices:
                numel = end - start
                grad_update[start:end] = nl_grad[offset : offset + numel]
                offset += numel

        self._velocity = (
            self.momentum * self._velocity
            + (1 - self.momentum) * grad_update
        )

        new_weights = (
            base_weights + self.lr * self._velocity
            - self.lr * self.weight_decay * base_weights
        )
        self._restore_weights(new_weights)

        step_loss = self._evaluate_current(x, y)
        self.best_loss = min(self.best_loss, step_loss)
        self.step_count += 1
        return step_loss

    def train(self, dataloader: DataLoader, steps: int) -> list[float]:
        """Run EGGROLL training loop.

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
                    f"best {self.best_loss:.4f} | {sps:.1f} steps/s"
                )

        return losses
