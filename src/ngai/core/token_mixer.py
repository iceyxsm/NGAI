"""RWKV-style gated recurrence token mixer for NGAI.

Replaces self-attention with O(n) recurrent token mixing.
No attention matrix is computed. Each token updates a fixed-size
hidden state via gated recurrence, using only element-wise operations.
"""

import torch
import torch.nn as nn
from torch import Tensor

from ngai.core.linear import TernaryLinear


class GatedRecurrence(nn.Module):
    """Gated recurrent token mixer inspired by RWKV.

    Processes a sequence token-by-token, maintaining a hidden state
    that summarizes history. Uses three gates:
    - Receptance (r): how much of the current input to accept
    - Key (k): what information to write to state
    - Value (v): the actual content to store

    The state decays over time via a learned decay factor,
    allowing the model to forget old information.

    Args:
        dim: Hidden dimension size.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.receptance = TernaryLinear(dim, dim)
        self.key = TernaryLinear(dim, dim)
        self.value = TernaryLinear(dim, dim)
        self.output = TernaryLinear(dim, dim)
        self.decay = nn.Parameter(torch.zeros(dim))
        self.bonus = nn.Parameter(torch.zeros(dim))

    CHUNK_SIZE = 16

    def forward(
        self,
        x: Tensor,
        state: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Process a sequence through chunked gated recurrence.

        Processes tokens in chunks for better GPU utilization while
        maintaining numerical stability. Each chunk runs the recurrence
        sequentially but the linear projections are batched.

        Args:
            x: Input tensor of shape (batch, seq_len, dim).
            state: Previous hidden state of shape (batch, dim).
                   If None, initializes to zeros.

        Returns:
            Tuple of (output, final_state):
            - output: shape (batch, seq_len, dim)
            - final_state: shape (batch, dim) for next call
        """
        batch, seq_len, _ = x.shape

        if state is None:
            state = torch.zeros(batch, self.dim, device=x.device, dtype=x.dtype)

        r = torch.sigmoid(self.receptance(x))
        k = self.key(x)
        v = self.value(x)

        w = torch.exp(-torch.exp(self.decay))
        bonus_scale = torch.exp(self.bonus)
        kv = (1.0 + bonus_scale) * k * v

        if seq_len == 1:
            state = w * state + kv[:, 0]
            output = self.output(r * state.unsqueeze(1))
            return output, state

        # Use CUDA kernel if available, else Python fallback
        if kv.is_cuda:
            try:
                from ngai.core.recurrence_cuda import cuda_recurrence
                states = cuda_recurrence(kv, w, state)
                output = self.output(r * states)
                return output, states[:, -1]
            except ImportError:
                pass

        # Python fallback for CPU
        all_states = []
        for t in range(seq_len):
            state = w * state + kv[:, t]
            all_states.append(state)

        states = torch.stack(all_states, dim=1)
        output = self.output(r * states)
        return output, state
