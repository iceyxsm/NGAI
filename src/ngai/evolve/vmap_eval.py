"""Vectorized population evaluation using torch.func.vmap.

Instead of evaluating variants one at a time (64 serial forward passes),
vmap vectorizes the forward pass across all variants simultaneously.
This saturates the GPU with a single kernel launch.

The trick: torch.func.functional_call lets us call a model with
arbitrary parameter dicts. vmap then vectorizes this across a batch
of parameter dicts — one per variant.

Expected speedup: 5-15x over serial evaluation.
"""

import torch
import torch.nn as nn
from torch import Tensor
from torch.func import functional_call, vmap


class VmapPopulationEvaluator:
    """Evaluate all population variants in parallel using vmap.

    Args:
        model: The NGAI model (used as template for architecture).
        device: Device to evaluate on.
    """

    def __init__(self, model: nn.Module, device: torch.device) -> None:
        self.model = model
        self.device = device
        self._param_names: list[str] = []
        self._param_shapes: list[tuple[int, ...]] = []
        self._param_numels: list[int] = []
        self._total_params = 0
        self._buf_names: list[str] = []
        self._cache_param_info()

    def _cache_param_info(self) -> None:
        """Cache parameter metadata for fast flat->dict conversion."""
        self._param_names = []
        self._param_shapes = []
        self._param_numels = []
        for name, p in self.model.named_parameters():
            self._param_names.append(name)
            self._param_shapes.append(tuple(p.shape))
            self._param_numels.append(p.numel())
        self._total_params = sum(self._param_numels)
        self._buf_names = [name for name, _ in self.model.named_buffers()]

    def _flat_to_param_dict(self, flat: Tensor) -> dict[str, Tensor]:
        """Convert flat weight vector to named parameter dict.

        Args:
            flat: Flat tensor of shape (total_params,).

        Returns:
            Dict mapping parameter names to shaped tensors.
        """
        params: dict[str, Tensor] = {}
        offset = 0
        for name, shape, numel in zip(
            self._param_names, self._param_shapes, self._param_numels,
            strict=True,
        ):
            params[name] = flat[offset : offset + numel].view(shape)
            offset += numel
        return params

    def _batch_flat_to_param_dict(
        self, flat_batch: Tensor,
    ) -> dict[str, Tensor]:
        """Convert batch of flat weights to batched parameter dict.

        Args:
            flat_batch: Tensor of shape (n_variants, total_params).

        Returns:
            Dict mapping param names to tensors of shape (n_variants, *shape).
        """
        params: dict[str, Tensor] = {}
        offset = 0
        for name, shape, numel in zip(
            self._param_names, self._param_shapes, self._param_numels,
            strict=True,
        ):
            chunk = flat_batch[:, offset : offset + numel]
            params[name] = chunk.view(-1, *shape)
            offset += numel
        return params

    def _get_buffers_dict(self) -> dict[str, Tensor]:
        """Get current model buffers as a dict."""
        return dict(self.model.named_buffers())

    @torch.no_grad()
    def evaluate_population_vmap(
        self,
        flat_variants: Tensor,
        x: Tensor,
        y: Tensor,
    ) -> Tensor:
        """Evaluate all variants in parallel using vmap.

        Uses torch.func.functional_call + vmap to vectorize the
        forward pass across all weight variants simultaneously.

        Args:
            flat_variants: Weight variants of shape (n_variants, total_params).
            x: Input of shape (batch, seq_len).
            y: Target of shape (batch, seq_len).

        Returns:
            Losses of shape (n_variants,).
        """
        batched_params = self._batch_flat_to_param_dict(flat_variants)
        buffers = self._get_buffers_dict()

        def single_forward(params: dict[str, Tensor]) -> Tensor:
            """Forward pass with one set of parameters."""
            result = functional_call(self.model, (params, buffers), (x,))
            logits = result[0] if isinstance(result, tuple) else result
            vocab = logits.shape[-1]
            return torch.nn.functional.cross_entropy(
                logits.view(-1, vocab), y.view(-1),
            )

        try:
            vectorized_forward = vmap(single_forward)
            return vectorized_forward(batched_params)
        except RuntimeError:
            return self._evaluate_population_serial(
                flat_variants, x, y,
            )

    @torch.no_grad()
    def _evaluate_population_serial(
        self,
        flat_variants: Tensor,
        x: Tensor,
        y: Tensor,
    ) -> Tensor:
        """Fallback serial evaluation if vmap fails.

        Args:
            flat_variants: Weight variants of shape (n_variants, total_params).
            x: Input of shape (batch, seq_len).
            y: Target of shape (batch, seq_len).

        Returns:
            Losses of shape (n_variants,).
        """
        n_variants = flat_variants.shape[0]
        losses = torch.zeros(n_variants, device=self.device)
        self.model.eval()

        for i in range(n_variants):
            params = self._flat_to_param_dict(flat_variants[i])
            buffers = self._get_buffers_dict()
            result = functional_call(
                self.model, (params, buffers), (x,),
            )
            logits = result[0] if isinstance(result, tuple) else result
            vocab = logits.shape[-1]
            losses[i] = torch.nn.functional.cross_entropy(
                logits.view(-1, vocab), y.view(-1),
            )
        return losses
