"""Multi-stream CUDA evaluation for parallel variant processing.

Uses multiple CUDA streams to evaluate several variants concurrently
on a single GPU. Each stream gets its own copy of the model and runs
an independent forward pass. The GPU schedules work from all streams
across its SMs simultaneously.

T4 has 40 SMs. A small model (dim=128) uses ~5 SMs per forward pass.
With 8 streams, we can run 8 variants in parallel = 8x speedup.

This is the correct way to parallelize ES on a single GPU:
- Same algorithm (weight perturbation + forward pass + loss)
- Same math (identical gradient estimate)
- Just concurrent execution instead of serial
"""

import copy

import torch
import torch.nn as nn
from torch import Tensor


class MultiStreamEvaluator:
    """Evaluate variants concurrently using multiple CUDA streams.

    Creates N copies of the model, each on its own CUDA stream.
    Variants are distributed round-robin across streams.

    Args:
        model: The NGAI model (template).
        n_streams: Number of concurrent streams.
        device: CUDA device.
    """

    DEFAULT_STREAMS = 8

    def __init__(
        self,
        model: nn.Module,
        n_streams: int = DEFAULT_STREAMS,
        device: torch.device | None = None,
    ) -> None:
        self.device = device or torch.device("cuda")
        self.n_streams = n_streams

        self._models = [
            copy.deepcopy(model).to(self.device) for _ in range(n_streams)
        ]
        self._streams = [
            torch.cuda.Stream(device=self.device) for _ in range(n_streams)
        ]

        self._param_shapes: list[tuple[int, ...]] = []
        self._param_numels: list[int] = []
        for p in model.parameters():
            self._param_shapes.append(tuple(p.shape))
            self._param_numels.append(p.numel())
        self._total_params = sum(self._param_numels)

    def _inject_weights_to_model(
        self, model_idx: int, flat: Tensor,
    ) -> None:
        """Copy flat weights into one model copy."""
        offset = 0
        for p, shape, numel in zip(
            self._models[model_idx].parameters(),
            self._param_shapes, self._param_numels,
            strict=True,
        ):
            p.data.copy_(flat[offset : offset + numel].view(shape))
            offset += numel

    @torch.no_grad()
    def evaluate_population(
        self,
        flat_variants: Tensor,
        x: Tensor,
        y: Tensor,
    ) -> Tensor:
        """Evaluate all variants using concurrent CUDA streams.

        Distributes variants across N streams. Each stream:
        1. Injects weights into its model copy
        2. Runs forward pass
        3. Computes loss

        All streams run concurrently on the GPU.

        Args:
            flat_variants: Shape (n_variants, total_params).
            x: Input of shape (batch, seq_len).
            y: Target of shape (batch, seq_len).

        Returns:
            Losses of shape (n_variants,).
        """
        n_variants = flat_variants.shape[0]
        losses = torch.empty(n_variants, device=self.device)
        vocab = -1

        for chunk_start in range(0, n_variants, self.n_streams):
            chunk_end = min(chunk_start + self.n_streams, n_variants)
            chunk_size = chunk_end - chunk_start

            for si in range(chunk_size):
                vi = chunk_start + si
                stream = self._streams[si]
                model = self._models[si]

                with torch.cuda.stream(stream):
                    self._inject_weights_to_model(si, flat_variants[vi])
                    model.eval()
                    logits = model(x)
                    if isinstance(logits, tuple):
                        logits = logits[0]
                    if vocab < 0:
                        vocab = logits.shape[-1]
                    losses[vi] = torch.nn.functional.cross_entropy(
                        logits.view(-1, vocab), y.view(-1),
                    )

            torch.cuda.synchronize()

        return losses
