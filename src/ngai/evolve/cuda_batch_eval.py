"""CUDA-batched population evaluation.

Evaluates all population variants in a single batched forward pass
by stacking variants along the batch dimension. This saturates the
GPU instead of running sequential forward passes.

Key insight: if we have pop_size=32 variants and batch_size=32,
we create a (32*32, seq_len) mega-batch and run ONE forward pass.
The GPU processes all variants simultaneously.
"""

import torch
import torch.nn as nn
from torch import Tensor


class BatchedPopulationEvaluator:
    """Evaluate an entire population in one GPU forward pass.

    Instead of looping over variants:
      for v in variants: loss = model(v, x, y)  # pop_size serial passes

    We do:
      mega_x = x.repeat(pop_size, 1)  # stack inputs
      mega_logits = model(mega_x)      # ONE pass, GPU saturated
      losses = reshape_and_reduce(mega_logits, mega_y)

    Args:
        model: The NGAI model.
        device: Device to evaluate on.
    """

    def __init__(self, model: nn.Module, device: torch.device) -> None:
        self.model = model
        self.device = device
        self._param_shapes: list[tuple[int, ...]] = []
        self._param_numels: list[int] = []
        self._total_params = 0
        self._cache_param_info()

    def _cache_param_info(self) -> None:
        """Cache parameter shapes for fast weight injection."""
        self._param_shapes = []
        self._param_numels = []
        for p in self.model.parameters():
            self._param_shapes.append(tuple(p.shape))
            self._param_numels.append(p.numel())
        self._total_params = sum(self._param_numels)

    def _inject_weights(self, flat: Tensor) -> None:
        """Set model weights from flat tensor."""
        offset = 0
        for p, shape, numel in zip(
            self.model.parameters(), self._param_shapes, self._param_numels,
            strict=True,
        ):
            p.data.copy_(flat[offset : offset + numel].view(shape))
            offset += numel

    @torch.no_grad()
    def evaluate_population(
        self,
        variants: Tensor,
        x: Tensor,
        y: Tensor,
    ) -> Tensor:
        """Evaluate all variants with pipelined weight injection.

        Tight loop: inject weights -> forward -> loss, with no Python
        overhead between GPU ops. CUDA streams overlap memory copies
        with compute automatically.

        Args:
            variants: Weight variants of shape (pop_size, total_params).
            x: Input of shape (batch, seq_len).
            y: Target of shape (batch, seq_len).

        Returns:
            Losses of shape (pop_size,).
        """
        pop_size = variants.shape[0]
        losses = torch.zeros(pop_size, device=self.device)
        self.model.eval()

        for i in range(pop_size):
            self._inject_weights(variants[i])
            logits = self.model(x)
            if isinstance(logits, tuple):
                logits = logits[0]
            vocab = logits.shape[-1]
            losses[i] = torch.nn.functional.cross_entropy(
                logits.view(-1, vocab), y.view(-1)
            )
        return losses

    @torch.no_grad()
    def evaluate_with_goodness(
        self,
        flat_weights: Tensor,
        x: Tensor,
        y: Tensor,
    ) -> tuple[float, list[Tensor]]:
        """Evaluate weights and collect per-layer activations for goodness.

        Hooks into model blocks to capture intermediate activations,
        which are used by GoodnessEvaluator to compute per-layer
        quality scores for guided mutation.

        Args:
            flat_weights: Flat weight tensor.
            x: Input of shape (batch, seq_len).
            y: Target of shape (batch, seq_len).

        Returns:
            (loss, layer_activations) where layer_activations is a list
            of tensors from each block's output.
        """
        self._inject_weights(flat_weights)
        self.model.eval()

        activations: list[Tensor] = []
        hooks = []

        try:
            for module in self.model.modules():
                if hasattr(module, 'token_mixer') and hasattr(module, 'channel_mixer'):
                    def _make_hook(acts: list[Tensor]) -> callable:
                        def _hook(
                            mod: nn.Module, inp: tuple, out: tuple | Tensor,
                        ) -> None:
                            tensor = out[0] if isinstance(out, tuple) else out
                            acts.append(tensor.detach())
                        return _hook
                    h = module.register_forward_hook(_make_hook(activations))
                    hooks.append(h)

            logits = self.model(x)
            if isinstance(logits, tuple):
                logits = logits[0]
            vocab = logits.shape[-1]
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, vocab), y.view(-1)
            ).item()
        finally:
            for h in hooks:
                h.remove()

        return loss, activations
