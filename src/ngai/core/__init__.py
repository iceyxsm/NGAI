"""Core engine and novel training algorithms.

Provides the fundamental building blocks for NGAI:
- Ternary weight quantization (BitNet b1.58 style)
- MatMul-free linear layers
- RWKV-style gated recurrence token mixer
"""

from ngai.core.linear import TernaryLinear
from ngai.core.ternary import ternary_quantize
from ngai.core.token_mixer import GatedRecurrence

__all__ = ["GatedRecurrence", "TernaryLinear", "ternary_quantize"]
