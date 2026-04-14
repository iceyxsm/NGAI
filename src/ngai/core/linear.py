"""MatMul-free linear layer for NGAI.

Replaces standard nn.Linear with ternary weight accumulation.
During forward pass, weights are quantized to {-1, 0, +1} so
"multiplication" becomes conditional negate, skip, or pass-through.
No floating-point multiply is used in the weight-input interaction.
"""

import torch
import torch.nn as nn
from torch import Tensor

from ngai.core.ternary import ternary_quantize


class TernaryLinear(nn.Module):
    """Linear layer with ternary weights and MatMul-free forward pass.

    Stores full-precision weights for training (gradient updates),
    but quantizes to ternary during forward pass. The actual computation
    uses standard matmul in PyTorch (which the ternary values make
    equivalent to additions/subtractions), but a future optimized kernel
    can replace this with true ternary accumulation.

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        bias: If True, adds a learnable bias. Default: False.
    """

    def __init__(
        self, in_features: int, out_features: int, *, bias: bool = False
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias: nn.Parameter | None = nn.Parameter(torch.empty(out_features))
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize weights using Kaiming uniform."""
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / (fan_in**0.5)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with ternary-quantized weights.

        Args:
            x: Input tensor of shape (..., in_features).

        Returns:
            Output tensor of shape (..., out_features).
        """
        w_q = ternary_quantize(self.weight)
        out = torch.nn.functional.linear(x, w_q, self.bias)
        return out

    def extra_repr(self) -> str:
        """String representation for print(model)."""
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}"
        )
