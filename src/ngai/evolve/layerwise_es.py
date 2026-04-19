"""Layer-wise Evolution Strategy with cached forward passes.

Perturbs one layer at a time and caches activations from unchanged
layers. When perturbing layer K, layers 0..K-1 produce identical
output for all perturbations, so we compute them once and reuse.

This cuts forward pass cost by ~50% on average:
- Perturbing layer 0: no cache benefit (all layers change)
- Perturbing layer 7: cache layers 0-6, only run layer 7 + head
- Average across 8 layers: ~50% compute saved
"""

import time

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader


class LayerwiseESTrainer:
    """ES trainer with activation caching for layer-wise perturbation.

    Args:
        model: The NGAI model to train.
        pop_size: Number of perturbation pairs per layer.
        sigma: Noise standard deviation.
        lr: Learning rate for weight updates.
        momentum: Momentum coefficient (per-layer velocity).
        weight_decay: L2 weight decay coefficient.
        device: Device to train on.
    """

    DEFAULT_POP = 32
    DEFAULT_SIGMA = 0.05
    DEFAULT_LR = 0.005
    DEFAULT_MOMENTUM = 0.9
    DEFAULT_WEIGHT_DECAY = 0.001
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
        self._block_map = self._map_groups_to_blocks()

    def _build_param_groups(self) -> list[dict]:
        """Group parameters by model block for layer-wise updates."""
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

    def _map_groups_to_blocks(self) -> dict[int, int]:
        """Map each param group to its block index (-1 for non-block)."""
        block_map: dict[int, int] = {}
        for gi, group in enumerate(self._param_groups):
            name = group["name"]
            block_map[gi] = -1
            if "blocks." in name:
                try:
                    block_map[gi] = int(
                        name.split("blocks.")[1].split(".")[0],
                    )
                except (ValueError, IndexError):
                    pass
        return block_map

    def _get_group_flat(self, group_idx: int) -> Tensor:
        """Flatten parameters for one group."""
        return torch.cat([
            p.data.view(-1) for p in self._param_groups[group_idx]["params"]
        ])

    def _set_group_flat(self, group_idx: int, flat: Tensor) -> None:
        """Set parameters for one group from flat tensor."""
        params = self._param_groups[group_idx]["params"]
        if len(params) == 1:
            params[0].data.copy_(flat.view(params[0].shape))
            return
        offset = 0
        for p in params:
            numel = p.numel()
            p.data.copy_(flat[offset : offset + numel].view(p.shape))
            offset += numel

    @torch.no_grad()
    def _forward_loss(self, x: Tensor, y: Tensor) -> Tensor:
        """Full forward pass returning loss as GPU tensor."""
        self.model.eval()
        logits = self.model(x)
        if isinstance(logits, tuple):
            logits = logits[0]
        vocab = logits.shape[-1]
        return torch.nn.functional.cross_entropy(
            logits.view(-1, vocab), y.view(-1),
        )

    @torch.no_grad()
    def _suffix_forward_loss(
        self, cached_x: Tensor, from_block: int, y: Tensor,
    ) -> Tensor:
        """Forward from cached activation through remaining blocks.

        Args:
            cached_x: Activation after blocks[0..from_block-1].
            from_block: First block to run.
            y: Target for loss computation.

        Returns:
            Loss as GPU tensor.
        """
        x = cached_x
        for i in range(from_block, len(self.model.blocks)):
            x, _ = self.model.blocks[i](x, None)
        x = self.model.norm(x)
        logits = self.model.head(x)
        if isinstance(logits, tuple):
            logits = logits[0]
        vocab = logits.shape[-1]
        return torch.nn.functional.cross_entropy(
            logits.view(-1, vocab), y.view(-1),
        )

    @torch.no_grad()
    def _compute_prefix(
        self, input_ids: Tensor, up_to_block: int,
    ) -> Tensor:
        """Run embedding + blocks[0..up_to_block-1], return activation.

        Args:
            input_ids: Token indices of shape (batch, seq_len).
            up_to_block: Stop before this block index.

        Returns:
            Cached activation tensor.
        """
        x = self.model.embedding(input_ids)
        for i in range(min(up_to_block, len(self.model.blocks))):
            x, _ = self.model.blocks[i](x, None)
        return x

    @staticmethod
    def _fitness_shaping(losses: Tensor) -> Tensor:
        """Vectorized rank-based fitness shaping."""
        n = losses.shape[0]
        ranks = torch.empty_like(losses)
        ranks[losses.argsort()] = torch.arange(
            n, dtype=losses.dtype, device=losses.device,
        )
        utilities = ranks / (n - 1) - 0.5
        return -utilities

    @torch.no_grad()
    def train_step(
        self, x: Tensor, y: Tensor, group_idx: int,
    ) -> float:
        """One ES step with activation caching.

        If the perturbed group belongs to block K, caches activations
        from blocks 0..K-1 and only re-runs blocks K..N per eval.

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
        scaled_noise = self.sigma * noise

        losses = torch.empty(self.pop_size * 2, device=self.device)

        block_idx = self._block_map[group_idx]
        use_cache = block_idx > 0
        cached_x = self._compute_prefix(x, block_idx) if use_cache else None

        for i in range(self.pop_size):
            self._set_group_flat(group_idx, base_weights + scaled_noise[i])
            if use_cache:
                losses[i * 2] = self._suffix_forward_loss(
                    cached_x, block_idx, y,
                )
            else:
                losses[i * 2] = self._forward_loss(x, y)

            self._set_group_flat(group_idx, base_weights - scaled_noise[i])
            if use_cache:
                losses[i * 2 + 1] = self._suffix_forward_loss(
                    cached_x, block_idx, y,
                )
            else:
                losses[i * 2 + 1] = self._forward_loss(x, y)

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

        step_loss = self._forward_loss(x, y).item()
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
        print(f"  Layer-wise ES: {self._n_groups} groups (cached fwd)")
        for i, (name, size) in enumerate(zip(group_names, group_sizes)):
            blk = self._block_map[i]
            tag = f"cache 0-{blk - 1}" if blk > 0 else "full"
            print(f"    [{i}] {name}: {size:,} params ({tag})")

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
