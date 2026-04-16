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

        Uses a vectorized parallel scan instead of a Python for-loop.
        For training (full sequences), computes all timesteps in parallel.
        For inference (single token), falls back to sequential update.

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

        # kv = k * v + bonus * k * v = (1 + bonus) * k * v
        kv = (1.0 + bonus_scale) * k * v

        if seq_len == 1:
            state = w * state + kv[:, 0]
            output = self.output(r * state.unsqueeze(1))
            return output, state

        # Vectorized parallel scan for the linear recurrence:
        # state_t = w * state_{t-1} + kv_t
        # This is equivalent to: state_t = sum_{i=0}^{t} w^{t-i} * kv_i
        # Compute using cumulative sum with exponential weights
        powers = torch.arange(seq_len, device=x.device, dtype=x.dtype)
        log_w = torch.log(w + 1e-8)
        # decay_matrix[t] = w^t (broadcast over dim)
        decay_powers = torch.exp(
            powers.unsqueeze(-1) * log_w.unsqueeze(0)
        )

        # Scale kv by inverse decay so cumsum gives correct result
        # kv_scaled[t] = kv[t] / w^t
        inv_decay = torch.exp(
            -powers.unsqueeze(-1) * log_w.unsqueeze(0)
        )
        kv_scaled = kv * inv_decay.unsqueeze(0)

        # Cumulative sum in the scaled domain
        cumsum = torch.cumsum(kv_scaled, dim=1)

        # Apply decay to get actual states: state[t] = w^t * cumsum[t]
        states = decay_powers.unsqueeze(0) * cumsum

        # Add contribution from initial state
        init_contrib = state.unsqueeze(1) * decay_powers.unsqueeze(0)
        states = states + init_contrib

        output = self.output(r * states)
        final_state = states[:, -1]
        return output, final_state
