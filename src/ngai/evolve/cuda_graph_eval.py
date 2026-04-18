"""CUDA graph-captured variant evaluation loop.

CUDA graphs capture a sequence of GPU operations and replay them
without Python overhead. We capture the inject->forward->loss
sequence once, then replay it 64 times with different weight data.

This eliminates the ~2ms Python overhead between each variant
evaluation, turning 64 * 2ms = 128ms of dead time into ~0.

Expected speedup: 2-4x over Python loop (on top of CUDA kernel).
"""

import torch
import torch.nn as nn
from torch import Tensor


class CUDAGraphEvaluator:
    """Evaluate population variants using CUDA graph replay.

    Captures the forward pass + loss computation as a CUDA graph,
    then replays it for each variant with only a memcpy to swap weights.

    Args:
        model: The NGAI model.
        device: CUDA device.
    """

    WARMUP_ITERS = 3

    def __init__(self, model: nn.Module, device: torch.device) -> None:
        self.model = model
        self.device = device
        self._param_shapes: list[tuple[int, ...]] = []
        self._param_numels: list[int] = []
        self._param_refs: list[nn.Parameter] = []
        self._total_params = 0
        self._graph: torch.cuda.CUDAGraph | None = None
        self._static_x: Tensor | None = None
        self._static_y: Tensor | None = None
        self._static_loss: Tensor | None = None
        self._captured = False
        self._cache_params()

    def _cache_params(self) -> None:
        """Cache parameter references for fast injection."""
        for p in self.model.parameters():
            self._param_shapes.append(tuple(p.shape))
            self._param_numels.append(p.numel())
            self._param_refs.append(p)
        self._total_params = sum(self._param_numels)

    def _inject_weights(self, flat: Tensor) -> None:
        """Copy flat weights into model parameters."""
        offset = 0
        for param, shape, numel in zip(
            self._param_refs, self._param_shapes, self._param_numels,
            strict=True,
        ):
            param.data.copy_(flat[offset : offset + numel].view(shape))
            offset += numel

    def _run_forward_loss(self) -> None:
        """Run forward pass and compute loss using static buffers."""
        self.model.eval()
        logits = self.model(self._static_x)
        if isinstance(logits, tuple):
            logits = logits[0]
        vocab = logits.shape[-1]
        self._static_loss.copy_(
            torch.nn.functional.cross_entropy(
                logits.view(-1, vocab), self._static_y.view(-1),
            ).unsqueeze(0)
        )

    def capture(self, x: Tensor, y: Tensor) -> None:
        """Capture the forward+loss as a CUDA graph.

        Must be called once before evaluate_population.
        The input shapes must match for all subsequent calls.

        Args:
            x: Input of shape (batch, seq_len).
            y: Target of shape (batch, seq_len).
        """
        if not x.is_cuda:
            return

        self._static_x = x.clone()
        self._static_y = y.clone()
        self._static_loss = torch.zeros(1, device=self.device)

        for _ in range(self.WARMUP_ITERS):
            self._run_forward_loss()

        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            self._run_forward_loss()

        self._captured = True

    @torch.no_grad()
    def evaluate_population(
        self,
        flat_variants: Tensor,
        x: Tensor,
        y: Tensor,
    ) -> Tensor:
        """Evaluate all variants using CUDA graph replay.

        Falls back to serial evaluation if graph capture failed
        or input is not on CUDA.

        Args:
            flat_variants: Shape (n_variants, total_params).
            x: Input of shape (batch, seq_len).
            y: Target of shape (batch, seq_len).

        Returns:
            Losses of shape (n_variants,).
        """
        n_variants = flat_variants.shape[0]
        losses = torch.empty(n_variants, device=self.device)

        if not self._captured or not x.is_cuda:
            return self._evaluate_serial(flat_variants, x, y)

        self._static_x.copy_(x)
        self._static_y.copy_(y)

        for i in range(n_variants):
            self._inject_weights(flat_variants[i])
            self._graph.replay()
            losses[i] = self._static_loss.item()

        return losses

    @torch.no_grad()
    def _evaluate_serial(
        self,
        flat_variants: Tensor,
        x: Tensor,
        y: Tensor,
    ) -> Tensor:
        """Fallback serial evaluation without CUDA graphs."""
        n_variants = flat_variants.shape[0]
        losses = torch.empty(n_variants, device=self.device)
        self.model.eval()

        for i in range(n_variants):
            self._inject_weights(flat_variants[i])
            logits = self.model(x)
            if isinstance(logits, tuple):
                logits = logits[0]
            vocab = logits.shape[-1]
            losses[i] = torch.nn.functional.cross_entropy(
                logits.view(-1, vocab), y.view(-1),
            )

        return losses
