"""NGAI Block: the fundamental repeating unit.

Combines a GatedRecurrence token mixer with a ternary feed-forward
network (channel mixer). This is the building block that gets stacked
to form the full model. Uses pre-norm (RMSNorm) for stability.
"""

import torch
import torch.nn as nn
from torch import Tensor

from ngai.core.linear import TernaryLinear
from ngai.core.token_mixer import GatedRecurrence


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Simpler and faster than LayerNorm — no mean subtraction, no bias.
    Just scale by the RMS of the input.

    Args:
        dim: Feature dimension to normalize over.
        eps: Small constant for numerical stability.
    """

    EPS_DEFAULT = 1e-6

    def __init__(self, dim: int, eps: float = EPS_DEFAULT) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        """Apply RMS normalization."""
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class ChannelMixer(nn.Module):
    """Ternary feed-forward network (channel mixer).

    Two ternary linear layers with SiLU gating, inspired by SwiGLU.
    Expands to 4x hidden dim, gates, then projects back down.

    Args:
        dim: Input/output dimension.
        expand_factor: FFN expansion ratio. Default: 4.
    """

    DEFAULT_EXPAND = 4

    def __init__(self, dim: int, expand_factor: int = DEFAULT_EXPAND) -> None:
        super().__init__()
        hidden = dim * expand_factor
        self.gate = TernaryLinear(dim, hidden)
        self.up = TernaryLinear(dim, hidden)
        self.down = TernaryLinear(hidden, dim)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with SiLU-gated ternary FFN."""
        return self.down(torch.nn.functional.silu(self.gate(x)) * self.up(x))


class NGAIBlock(nn.Module):
    """Single NGAI block: token mixer + channel mixer with residuals.

    Architecture per block:
    1. RMSNorm -> GatedRecurrence (token mixing) -> residual add
    2. RMSNorm -> ChannelMixer (channel mixing) -> residual add

    Args:
        dim: Hidden dimension size.
        expand_factor: FFN expansion ratio. Default: 4.
    """

    DEFAULT_EXPAND = 4

    def __init__(self, dim: int, expand_factor: int = DEFAULT_EXPAND) -> None:
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.token_mixer = GatedRecurrence(dim)
        self.norm2 = RMSNorm(dim)
        self.channel_mixer = ChannelMixer(dim, expand_factor)

    def forward(
        self, x: Tensor, state: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """Forward pass through one NGAI block.

        Args:
            x: Input of shape (batch, seq_len, dim).
            state: Recurrent state from previous call.

        Returns:
            Tuple of (output, new_state).
        """
        residual = x
        mixed, state = self.token_mixer(self.norm1(x), state)
        x = residual + mixed

        residual = x
        x = residual + self.channel_mixer(self.norm2(x))

        return x, state
