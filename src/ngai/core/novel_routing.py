"""Novel routing mechanisms for NGAI.

Three genuinely novel ideas: StateAwareRouter, AdaptiveRouter,
CrossLayerExpertPool.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from ngai.core.linear import TernaryLinear


class StateAwareRouter(nn.Module):
    """Router that uses both current token and recurrent hidden state.

    Routes using the concatenation of the current token representation
    and a projected recurrent state, allowing routing decisions to be
    informed by sequence-level context.

    Args:
        dim: Input dimension.
        n_experts: Total number of routed experts.
        top_k: Number of experts to activate per token.
    """

    def __init__(self, dim: int, n_experts: int, top_k: int) -> None:
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.gate = TernaryLinear(dim * 2, n_experts)
        self.state_proj = TernaryLinear(dim, dim)

    def forward(
        self, x: Tensor, state: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Route tokens using current input and recurrent state.

        Args:
            x: Input of shape (n_tokens, dim).
            state: Recurrent hidden state of shape (n_tokens, dim).
                If None, zeros are used.

        Returns:
            Tuple of (weights, indices, balance_loss).
        """
        if state is None:
            state = torch.zeros_like(x)

        state_context = self.state_proj(state)
        combined = torch.cat([x, state_context], dim=-1)
        logits = self.gate(combined)
        probs = F.softmax(logits, dim=-1)

        weights, indices = torch.topk(probs, self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)

        # Balance loss: encourage uniform expert utilization
        n_tokens = x.shape[0]
        expert_mask = F.one_hot(indices, self.n_experts).sum(dim=1).float()
        f = expert_mask.sum(dim=0) / n_tokens
        p = probs.mean(dim=0)
        balance_loss = self.n_experts * (f * p).sum()

        return weights, indices, balance_loss


class AdaptiveRouter(nn.Module):
    """Router that adapts the number of experts per token based on difficulty.

    Decides how many experts (from 1 to max_k) each token should use,
    based on a learned difficulty score. Easy tokens use fewer experts,
    hard tokens use more.

    Args:
        dim: Input dimension.
        n_experts: Total number of routed experts.
        max_k: Maximum number of experts to activate per token.
    """

    MIN_EXPERTS = 1
    RENORM_EPS = 1e-8

    def __init__(self, dim: int, n_experts: int, max_k: int) -> None:
        super().__init__()
        self.n_experts = n_experts
        self.max_k = max_k
        self.gate = TernaryLinear(dim, n_experts)
        self.difficulty = TernaryLinear(dim, 1)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Route tokens with adaptive expert count.

        Args:
            x: Input of shape (n_tokens, dim).

        Returns:
            Tuple of (weights, indices, balance_loss).
            Access self.last_avg_k for monitoring.
        """
        logits = self.gate(x)
        probs = F.softmax(logits, dim=-1)

        # Difficulty score: 0 = easy (fewer experts), 1 = hard (more experts)
        difficulty_score = torch.sigmoid(self.difficulty(x)).squeeze(-1)
        k_per_token = (
            (self.MIN_EXPERTS + difficulty_score * (self.max_k - self.MIN_EXPERTS))
            .round()
            .long()
            .clamp(self.MIN_EXPERTS, self.max_k)
        )

        # Select max_k experts, then mask inactive ones
        weights, indices = torch.topk(probs, self.max_k, dim=-1)
        positions = torch.arange(self.max_k, device=x.device).unsqueeze(0)
        active_mask = positions < k_per_token.unsqueeze(-1)
        weights = weights * active_mask.float()
        weights = weights / (weights.sum(-1, keepdim=True) + self.RENORM_EPS)

        # Balance loss: encourage uniform expert utilization
        n_tokens = x.shape[0]
        expert_mask = F.one_hot(indices, self.n_experts).sum(dim=1).float()
        f = expert_mask.sum(dim=0) / n_tokens
        p = probs.mean(dim=0)
        balance_loss = self.n_experts * (f * p).sum()

        self.last_avg_k = k_per_token.float().mean()

        return weights, indices, balance_loss


class CrossLayerExpertPool(nn.Module):
    """Shared pool of experts used across all layers.

    Instead of each layer owning its own experts, a single pool is
    shared. Each layer's router picks from this common pool, enabling
    cross-layer expert reuse and reducing total parameter count.

    Args:
        dim: Hidden dimension.
        n_experts: Number of experts in the shared pool.
        expand_factor: FFN expansion ratio per expert.
    """

    DEFAULT_EXPAND = 2

    def __init__(
        self, dim: int, n_experts: int, expand_factor: int = DEFAULT_EXPAND
    ) -> None:
        super().__init__()
        from ngai.core.block import ChannelMixer

        self.experts = nn.ModuleList(
            [ChannelMixer(dim, expand_factor) for _ in range(n_experts)]
        )
        self.n_experts = n_experts

    def forward(self, x: Tensor, weights: Tensor, indices: Tensor) -> Tensor:
        """Apply selected experts from the shared pool to input.

        Args:
            x: Input of shape (n_tokens, dim).
            weights: Expert weights of shape (n_tokens, top_k).
            indices: Expert indices of shape (n_tokens, top_k).

        Returns:
            Routed output of shape (n_tokens, dim).
        """
        routed_out = torch.zeros_like(x)
        top_k = indices.shape[-1]

        for k in range(top_k):
            expert_idx = indices[:, k]
            w = weights[:, k].unsqueeze(-1)
            for i, expert in enumerate(self.experts):
                mask = expert_idx == i
                if mask.any():
                    routed_out[mask] += w[mask] * expert(x[mask])

        return routed_out
