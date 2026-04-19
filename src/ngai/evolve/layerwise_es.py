"""Layer-wise Evolution Strategy trainer with multi-stream eval.

Instead of perturbing all parameters at once (which requires O(d)
population size for d-dimensional space), perturb one layer at a time.

Uses multiple CUDA streams for parallel evaluation: N model copies
each run a different perturbation concurrently, giving ~Nx speedup.
"""

import copy
import time

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader


class LayerwiseESTrainer:
    """ES trainer that perturbs one parameter group at a time.

    Cycles through model parameter groups (layers), applying ES
    updates to each in turn. Uses CUDA streams for parallel eval.

    Args:
        model: The NGAI model to train.
        pop_size: Number of perturbation pairs per layer.
        sigma: Noise standard deviation.
        lr: Learning rate for weight updates.
        momentum: Momentum coefficient (per-layer velocity).
        weight_decay: L2 weight decay coefficient.
        n_streams: Number of concurrent CUDA streams for eval.
        device: Device to train on.
    """

    DEFAULT_POP = 32
    DEFAULT_SIGMA = 0.05
    DEFAULT_LR = 0.005
    DEFAULT_MOMENTUM = 0.9
    DEFAULT_WEIGHT_DECAY = 0.001
    DEFAULT_STREAMS = 4
    LOG_EVERY = 50
    MIN_GROUP_PARAMS = 1024

    def __init__(
        self,
        model: nn.Module,
        pop_size: int = DEFAULT_POP,
        sigma: float = DEFAULT_SIGMA,
        lr: float = DEFAULT_LR,
        momentum: float = DEFAULT_MOMENTUM,
        weight_decay: float = DEFAULT_WEIGHT_DECAY,
        n_streams: int = DEFAULT_STREAMS,
        device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.device = device or torch.device("cpu")
        self.pop_size = pop_size
        self.sigma = sigma
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.best_loss = float("inf")
        self.step_count = 0
        self.total_steps = 0

        self._param_groups = self._build_param_groups()
        self._n_groups = len(self._param_groups)
        self._velocities = [
            torch.zeros(g["numel"], device=self.device)
            for g in self._param_groups
        ]
        self._total_params = sum(p.numel() for p in model.parameters())

        self._n_streams = n_streams if self.device.type == "cuda" else 1
        self._init_streams()

    def _init_streams(self) -> None:
        """Create model copies and CUDA streams for parallel eval."""
        if self._n_streams <= 1:
            self._models = [self.model]
            self._streams = [None]
            return
        self._models = [
            copy.deepcopy(self.model).to(self.device)
            for _ in range(self._n_streams)
        ]
        self._streams = [
            torch.cuda.Stream(device=self.device)
            for _ in range(self._n_streams)
        ]

    def _sync_model_to_copies(self) -> None:
        """Copy current model weights to all stream copies."""
        if self._n_streams <= 1:
            return
        state = self.model.state_dict()
        for m in self._models:
            m.load_state_dict(state)

    def _build_param_groups(self) -> list[dict]:
        """Group parameters by model block for layer-wise updates.

        Returns:
            List of dicts with 'params', 'numel', 'name' for each group.
        """
        groups: list[dict] = []
        for name, module in self.model.named_modules():
            direct_params = list(module.parameters(recurse=False))
            if not direct_params:
                continue
            numel = sum(p.numel() for p in direct_params)
            if numel < self.MIN_GROUP_PARAMS:
                continue
            groups.append({
                "name": name or "root",
                "params": direct_params,
                "numel": numel,
            })
        if not groups:
            all_params = list(self.model.parameters())
            groups.append({
                "name": "all",
                "params": all_params,
                "numel": sum(p.numel() for p in all_params),
            })
        return groups

    def _get_group_flat(self, group_idx: int) -> Tensor:
        """Flatten parameters for one group."""
        return torch.cat([
            p.data.view(-1) for p in self._param_groups[group_idx]["params"]
        ])

    def _set_group_flat(self, group_idx: int, flat: Tensor) -> None:
        """Set parameters for one group from flat tensor."""
        offset = 0
        for p in self._param_groups[group_idx]["params"]:
            numel = p.numel()
            p.data.copy_(flat[offset : offset + numel].view(p.shape))
            offset += numel

    def _find_copy_params(
        self, model_copy: nn.Module, group_idx: int,
    ) -> list[nn.Parameter]:
        """Find matching parameter group in a model copy."""
        group = self._param_groups[group_idx]
        target_name = group["name"]
        for name, module in model_copy.named_modules():
            if (name or "root") == target_name:
                return list(module.parameters(recurse=False))
        return []

    def _set_copy_group(
        self, copy_idx: int, group_idx: int, flat: Tensor,
    ) -> None:
        """Set one group's params on a model copy."""
        params = self._find_copy_params(
            self._models[copy_idx], group_idx,
        )
        offset = 0
        for p in params:
            numel = p.numel()
            p.data.copy_(flat[offset : offset + numel].view(p.shape))
            offset += numel

    @torch.no_grad()
    def _eval_on_copy(
        self, copy_idx: int, x: Tensor, y: Tensor,
    ) -> Tensor:
        """Run forward pass on a model copy, return loss as tensor."""
        model = self._models[copy_idx]
        model.eval()
        logits = model(x)
        if isinstance(logits, tuple):
            logits = logits[0]
        vocab = logits.shape[-1]
        return torch.nn.functional.cross_entropy(
            logits.view(-1, vocab), y.view(-1),
        )

    @torch.no_grad()
    def _evaluate(self, x: Tensor, y: Tensor) -> float:
        """Evaluate current model weights on a single batch."""
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

    @torch.no_grad()
    def _eval_batch_parallel(
        self,
        group_idx: int,
        base_weights: Tensor,
        noise: Tensor,
        x: Tensor,
        y: Tensor,
    ) -> Tensor:
        """Evaluate all perturbations using parallel CUDA streams.

        Processes perturbations in chunks of n_streams, running
        concurrent forward passes on separate model copies.

        Args:
            group_idx: Which parameter group is being perturbed.
            base_weights: Flat weights for the target group.
            noise: Noise matrix of shape (pop_size, n_params).
            x: Input batch.
            y: Target batch.

        Returns:
            Losses of shape (pop_size * 2,).
        """
        losses = torch.zeros(self.pop_size * 2, device=self.device)
        evals = []
        for i in range(self.pop_size):
            evals.append(("pos", i, base_weights + self.sigma * noise[i]))
            evals.append(("neg", i, base_weights - self.sigma * noise[i]))

        for chunk_start in range(0, len(evals), self._n_streams):
            chunk = evals[chunk_start:chunk_start + self._n_streams]
            for si, (_, _, perturbed) in enumerate(chunk):
                stream = self._streams[si]
                with torch.cuda.stream(stream):
                    self._set_copy_group(si, group_idx, perturbed)
                    losses[chunk_start + si] = self._eval_on_copy(
                        si, x, y,
                    )
            torch.cuda.synchronize()

        return losses

    @torch.no_grad()
    def _eval_batch_serial(
        self,
        group_idx: int,
        base_weights: Tensor,
        noise: Tensor,
        x: Tensor,
        y: Tensor,
    ) -> Tensor:
        """Evaluate all perturbations serially (CPU fallback).

        Args:
            group_idx: Which parameter group is being perturbed.
            base_weights: Flat weights for the target group.
            noise: Noise matrix of shape (pop_size, n_params).
            x: Input batch.
            y: Target batch.

        Returns:
            Losses of shape (pop_size * 2,).
        """
        losses = torch.zeros(self.pop_size * 2, device=self.device)
        for i in range(self.pop_size):
            self._set_group_flat(
                group_idx, base_weights + self.sigma * noise[i],
            )
            losses[i * 2] = self._evaluate(x, y)
            self._set_group_flat(
                group_idx, base_weights - self.sigma * noise[i],
            )
            losses[i * 2 + 1] = self._evaluate(x, y)
        self._set_group_flat(group_idx, base_weights)
        return losses

    @torch.no_grad()
    def train_step(
        self, x: Tensor, y: Tensor, group_idx: int,
    ) -> float:
        """One ES step on a single parameter group.

        Args:
            x: Input batch of shape (batch, seq_len).
            y: Target batch of shape (batch, seq_len).
            group_idx: Which parameter group to update.

        Returns:
            Loss after the update.
        """
        base_weights = self._get_group_flat(group_idx)
        noise = torch.randn(
            self.pop_size, self._param_groups[group_idx]["numel"],
            device=self.device,
        )

        if self._n_streams > 1:
            self._sync_model_to_copies()
            losses = self._eval_batch_parallel(
                group_idx, base_weights, noise, x, y,
            )
        else:
            losses = self._eval_batch_serial(
                group_idx, base_weights, noise, x, y,
            )

        self._set_group_flat(group_idx, base_weights)

        shaped = self._fitness_shaping(losses)
        pos_shaped = shaped[0::2]
        neg_shaped = shaped[1::2]
        diffs = (pos_shaped - neg_shaped).unsqueeze(1)
        grad_estimate = (diffs * noise).sum(dim=0) / (
            self.pop_size * self.sigma
        )

        vel = self._velocities[group_idx]
        vel.mul_(self.momentum).add_(grad_estimate, alpha=1 - self.momentum)

        new_weights = (
            base_weights + self.lr * vel
            - self.lr * self.weight_decay * base_weights
        )
        self._set_group_flat(group_idx, new_weights)

        step_loss = self._evaluate(x, y)
        self.best_loss = min(self.best_loss, step_loss)
        self.step_count += 1
        return step_loss

    def train(self, dataloader: DataLoader, steps: int) -> list[float]:
        """Run layer-wise ES training loop.

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

        group_names = [g["name"] for g in self._param_groups]
        group_sizes = [g["numel"] for g in self._param_groups]
        print(f"  Layer-wise ES: {self._n_groups} groups, "
              f"{self._n_streams} CUDA streams")
        for i, (name, size) in enumerate(zip(group_names, group_sizes)):
            print(f"    [{i}] {name}: {size:,} params")

        for step in range(steps):
            try:
                x, y = next(loader_iter)
            except StopIteration:
                loader_iter = iter(dataloader)
                x, y = next(loader_iter)
            x, y = x.to(self.device), y.to(self.device)

            group_idx = step % self._n_groups
            loss = self.train_step(x, y, group_idx)
            losses.append(loss)

            if (step + 1) % self.LOG_EVERY == 0:
                avg = sum(losses[-self.LOG_EVERY:]) / self.LOG_EVERY
                elapsed = time.perf_counter() - t0
                sps = (step + 1) / elapsed
                gname = self._param_groups[group_idx]["name"]
                print(
                    f"  step {step + 1:>5} | loss {avg:.4f} | "
                    f"best {self.best_loss:.4f} | {sps:.1f} steps/s | "
                    f"layer {gname}"
                )

        return losses
