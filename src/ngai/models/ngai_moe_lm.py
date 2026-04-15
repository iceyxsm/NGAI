"""NGAI MoE Language Model.

Stacks NGAIMoEBlocks with token embedding and output head.
Uses sparse activation — only a subset of experts compute per token,
giving the model more total capacity than active parameters.
"""

import torch.nn as nn
from torch import Tensor

from ngai.core.block import NGAIMoEBlock, RMSNorm
from ngai.core.linear import TernaryLinear


class NGAIMoELanguageModel(nn.Module):
    """Causal language model with MoE sparse activation.

    Args:
        vocab_size: Number of tokens in vocabulary.
        dim: Hidden dimension size.
        n_layers: Number of MoE blocks to stack.
        n_shared: Shared experts per block.
        n_routed: Routed experts per block.
        top_k: Experts activated per token.
        expand_factor: FFN expansion ratio per expert.
    """

    DEFAULT_EXPAND = 2
    BALANCE_LOSS_WEIGHT = 0.01

    def __init__(
        self,
        vocab_size: int,
        dim: int,
        n_layers: int,
        n_shared: int = 1,
        n_routed: int = 8,
        top_k: int = 2,
        expand_factor: int = DEFAULT_EXPAND,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([
            NGAIMoEBlock(dim, n_shared, n_routed, top_k, expand_factor)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(dim)
        self.head = TernaryLinear(dim, vocab_size)

    def forward(
        self,
        input_ids: Tensor,
        states: list[Tensor] | None = None,
    ) -> tuple[Tensor, list[Tensor], Tensor]:
        """Forward pass for MoE language modeling.

        Args:
            input_ids: Token indices of shape (batch, seq_len).
            states: List of recurrent states, one per block.

        Returns:
            (logits, new_states, total_balance_loss)
        """
        if states is None:
            states = [None] * self.n_layers

        x = self.embedding(input_ids)
        new_states: list[Tensor] = []
        total_balance_loss = 0.0

        for block, state in zip(self.blocks, states, strict=True):
            x, s, bl = block(x, state)
            new_states.append(s)
            total_balance_loss = total_balance_loss + bl

        x = self.norm(x)
        logits = self.head(x)
        return logits, new_states, total_balance_loss * self.BALANCE_LOSS_WEIGHT

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_active_parameters(self) -> int:
        """Estimate active parameters per token (shared + top_k routed)."""
        if not self.blocks:
            return 0
        block = self.blocks[0]
        mixer = block.channel_mixer
        shared_params = sum(
            p.numel() for e in mixer.shared_experts for p in e.parameters()
        )
        per_expert = sum(
            p.numel() for p in mixer.routed_experts[0].parameters()
        )
        active_routed = per_expert * mixer.top_k
        mixer_active = shared_params + active_routed
        token_mixer_params = sum(
            p.numel() for p in block.token_mixer.parameters()
        )
        norm_params = sum(
            p.numel() for p in block.norm1.parameters()
        ) + sum(p.numel() for p in block.norm2.parameters())
        per_block = token_mixer_params + norm_params + mixer_active
        embed_params = sum(p.numel() for p in self.embedding.parameters())
        head_params = sum(p.numel() for p in self.head.parameters())
        return per_block * self.n_layers + embed_params + head_params
