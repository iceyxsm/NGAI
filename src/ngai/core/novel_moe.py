"""MoE channel mixer variants using novel routers.

StateAwareMoE uses recurrent state for routing decisions.
AdaptiveMoE adapts the number of experts per token based on difficulty.
Both follow the same interface pattern as MoEChannelMixer.
"""

import torch
import torch.nn as nn
from torch import Tensor

from ngai.core.block import ChannelMixer
from ngai.core.novel_routing import AdaptiveRouter, StateAwareRouter


class StateAwareMoE(nn.Module):
    """MoE channel mixer that routes using both token and recurrent state.

    Same structure as MoEChannelMixer but uses StateAwareRouter,
    allowing routing decisions to be informed by sequence-level context.

    Args:
        dim: Hidden dimension.
        n_shared: Number of always-active shared experts.
        n_routed: Number of routed experts.
        top_k: Routed experts to activate per token.
        expand_factor: FFN expansion ratio per expert.
    """

    DEFAULT_EXPAND = 2

    def __init__(
        self,
        dim: int,
        n_shared: int,
        n_routed: int,
        top_k: int,
        expand_factor: int = DEFAULT_EXPAND,
    ) -> None:
        super().__init__()
        self.n_shared = n_shared
        self.n_routed = n_routed
        self.top_k = top_k

        self.shared_experts = nn.ModuleList(
            [ChannelMixer(dim, expand_factor) for _ in range(n_shared)]
        )
        self.routed_experts = nn.ModuleList(
            [ChannelMixer(dim, expand_factor) for _ in range(n_routed)]
        )
        self.router = StateAwareRouter(dim, n_routed, top_k)

    def forward(
        self, x: Tensor, state: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """Forward pass through state-aware MoE channel mixer.

        Args:
            x: Input of shape (batch, seq_len, dim).
            state: Recurrent state of shape (batch, dim) or None.
                Broadcast to all tokens for routing context.

        Returns:
            (output, balance_loss)
        """
        batch, seq_len, dim = x.shape
        flat = x.reshape(-1, dim)

        # Broadcast state to all tokens: (batch, dim) -> (batch*seq_len, dim)
        if state is not None:
            flat_state = state.unsqueeze(1).expand(-1, seq_len, -1).reshape(-1, dim)
        else:
            flat_state = None

        # Shared experts: always active
        shared_out = torch.zeros_like(flat)
        for expert in self.shared_experts:
            shared_out = shared_out + expert(flat)

        # Router selects top-k routed experts per token
        weights, indices, balance_loss = self.router(flat, flat_state)

        # Compute routed expert outputs
        routed_out = torch.zeros_like(flat)
        for k in range(self.top_k):
            expert_idx = indices[:, k]
            w = weights[:, k].unsqueeze(-1)
            for i, expert in enumerate(self.routed_experts):
                mask = expert_idx == i
                if mask.any():
                    routed_out[mask] += w[mask] * expert(flat[mask])

        output = shared_out + routed_out
        return output.reshape(batch, seq_len, dim), balance_loss


class AdaptiveMoE(nn.Module):
    """MoE channel mixer that adapts expert count per token.

    Uses AdaptiveRouter to decide how many experts (1 to max_k) each
    token needs. Easy tokens use fewer experts, hard tokens use more.

    Args:
        dim: Hidden dimension.
        n_shared: Number of always-active shared experts.
        n_routed: Number of routed experts.
        max_k: Maximum routed experts to activate per token.
        expand_factor: FFN expansion ratio per expert.
    """

    DEFAULT_EXPAND = 2

    def __init__(
        self,
        dim: int,
        n_shared: int,
        n_routed: int,
        max_k: int,
        expand_factor: int = DEFAULT_EXPAND,
    ) -> None:
        super().__init__()
        self.n_shared = n_shared
        self.n_routed = n_routed
        self.max_k = max_k

        self.shared_experts = nn.ModuleList(
            [ChannelMixer(dim, expand_factor) for _ in range(n_shared)]
        )
        self.routed_experts = nn.ModuleList(
            [ChannelMixer(dim, expand_factor) for _ in range(n_routed)]
        )
        self.router = AdaptiveRouter(dim, n_routed, max_k)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Forward pass through adaptive MoE channel mixer.

        Args:
            x: Input of shape (batch, seq_len, dim).

        Returns:
            (output, balance_loss, avg_k) where avg_k is the mean
            number of experts used per token (useful for monitoring).
        """
        batch, seq_len, dim = x.shape
        flat = x.reshape(-1, dim)

        # Shared experts: always active
        shared_out = torch.zeros_like(flat)
        for expert in self.shared_experts:
            shared_out = shared_out + expert(flat)

        # Router selects adaptive number of routed experts per token
        weights, indices, balance_loss, avg_k = self.router(flat)

        # Compute routed expert outputs
        routed_out = torch.zeros_like(flat)
        for k in range(self.max_k):
            expert_idx = indices[:, k]
            w = weights[:, k].unsqueeze(-1)
            for i, expert in enumerate(self.routed_experts):
                mask = expert_idx == i
                if mask.any():
                    routed_out[mask] += w[mask] * expert(flat[mask])

        output = shared_out + routed_out
        return output.reshape(batch, seq_len, dim), balance_loss, avg_k
