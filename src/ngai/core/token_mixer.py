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

    def forward(
        self,
        x: Tensor,
        state: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Process a sequence through gated recurrence.

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

        outputs = []
        for t in range(seq_len):
            k_t = k[:, t]
            v_t = v[:, t]
            bonus = torch.exp(self.bonus) * k_t * v_t
            state = w * state + k_t * v_t + bonus
            outputs.append(r[:, t] * state)

        output = torch.stack(outputs, dim=1)
        output = self.output(output)
        return output, state
