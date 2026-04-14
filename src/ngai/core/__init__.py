"""Core engine and novel training algorithms.

Provides the fundamental building blocks for NGAI:
- Ternary weight quantization (BitNet b1.58 style)
- MatMul-free linear layers
- RWKV-style gated recurrence token mixer
- Full NGAI block (token mixer + channel mixer)
"""

from ngai.core.block import ChannelMixer, NGAIBlock, RMSNorm
from ngai.core.linear import TernaryLinear
from ngai.core.ternary import ternary_quantize
from ngai.core.token_mixer import GatedRecurrence

__all__ = [
    "ChannelMixer",
    "GatedRecurrence",
    "NGAIBlock",
    "RMSNorm",
    "TernaryLinear",
    "ternary_quantize",
]
