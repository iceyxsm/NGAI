"""Mixture of Experts for NGAI.

DeepSeek-style MoE with shared + routed experts and fine-grained
expert segmentation. The router selects top-k experts per token,
while shared experts are always active (capturing common knowledge).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from ngai.core.block import ChannelMixer
from ngai.core.linear import TernaryLinear


class ExpertRouter(nn.Module):
    """Top-k router that selects which experts process each token.

    Args:
        dim: Input dimension.
        n_experts: Total number of routed experts.
        top_k: Number of experts to activate per token.
    """

    def __init__(self, dim: int, n_experts: int, top_k: int) -> None:
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.gate = TernaryLinear(dim, n_experts)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Route tokens to experts.

        Args:
            x: Input of shape (n_tokens, dim).

        Returns:
            (weights, indices, balance_loss)
        """
        logits = self.gate(x)
        probs = F.softmax(logits, dim=-1)
        weights, indices = torch.topk(probs, self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)

        n_tokens = x.shape[0]
        expert_mask = F.one_hot(indices, self.n_experts).sum(dim=1).float()
        f = expert_mask.sum(dim=0) / n_tokens
        p = probs.mean(dim=0)
        balance_loss = self.n_experts * (f * p).sum()

        return weights, indices, balance_loss


class MoEChannelMixer(nn.Module):
    """Mixture of Experts channel mixer with shared + routed experts.

    DeepSeek-style: shared experts always active, routed experts
    selected per-token. Output = shared + weighted_sum(routed).

    Args:
        dim: Hidden dimension.
        n_shared: Number of always-active shared experts.
        n_routed: Number of routed experts.
        top_k: Routed experts to activate per token.
        expand_factor: FFN expansion ratio per expert.
    """

    DEFAULT_EXPAND = 4

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
        self.router = ExpertRouter(dim, n_routed, top_k)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Forward pass through MoE channel mixer.

        Args:
            x: Input of shape (batch, seq_len, dim).

        Returns:
            (output, balance_loss)
        """
        batch, seq_len, dim = x.shape
        flat = x.reshape(-1, dim)

        # Shared experts: always active
        shared_out = torch.zeros_like(flat)
        for expert in self.shared_experts:
            shared_out = shared_out + expert(flat)

        # Router selects top-k routed experts per token
        weights, indices, balance_loss = self.router(flat)

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
