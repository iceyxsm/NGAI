"""Ternary weight quantization for NGAI.

Implements BitNet b1.58-style absmean quantization: full-precision weights
are quantized to {-1, 0, +1} during the forward pass, with straight-through
estimator (STE) for gradient flow during training.
"""

import torch
from torch import Tensor


class TernaryQuantize(torch.autograd.Function):
    """Quantize weights to {-1, 0, +1} using absmean scaling.

    Forward: scale by mean absolute value, round to nearest ternary value.
    Backward: straight-through estimator (pass gradients unchanged).
    """

    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, w: Tensor) -> Tensor:
        gamma = w.abs().mean()
        w_scaled = w / (gamma + 1e-8)
        w_ternary = w_scaled.clamp(-1, 1).round()
        return w_ternary * gamma

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: Tensor
    ) -> Tensor:
        return grad_output


def ternary_quantize(w: Tensor) -> Tensor:
    """Quantize a weight tensor to ternary {-1, 0, +1} scaled by absmean.

    Args:
        w: Full-precision weight tensor of any shape.

    Returns:
        Ternary-quantized weight tensor with same shape.
    """
    return TernaryQuantize.apply(w)
