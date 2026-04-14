"""NGAI Language Model.

Stacks N NGAI blocks with token embedding and output head
to form a complete causal language model. All weights are
ternary-quantized during forward pass.
"""

import torch.nn as nn
from torch import Tensor

from ngai.core.block import NGAIBlock, RMSNorm
from ngai.core.linear import TernaryLinear


class NGAILanguageModel(nn.Module):
    """Causal language model built from NGAI blocks.

    Args:
        vocab_size: Number of tokens in vocabulary.
        dim: Hidden dimension size.
        n_layers: Number of NGAI blocks to stack.
        expand_factor: FFN expansion ratio.
    """

    DEFAULT_EXPAND = 4

    def __init__(
        self,
        vocab_size: int,
        dim: int,
        n_layers: int,
        expand_factor: int = DEFAULT_EXPAND,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList(
            [NGAIBlock(dim, expand_factor) for _ in range(n_layers)]
        )
        self.norm = RMSNorm(dim)
        self.head = TernaryLinear(dim, vocab_size)

    def forward(
        self,
        input_ids: Tensor,
        states: list[Tensor] | None = None,
    ) -> tuple[Tensor, list[Tensor]]:
        """Forward pass for language modeling.

        Args:
            input_ids: Token indices of shape (batch, seq_len).
            states: List of recurrent states, one per block.

        Returns:
            Tuple of (logits, new_states):
            - logits: shape (batch, seq_len, vocab_size)
            - new_states: list of tensors, one per block
        """
        if states is None:
            states = [None] * self.n_layers

        x = self.embedding(input_ids)
        new_states: list[Tensor] = []

        for block, state in zip(self.blocks, states, strict=True):
            x, s = block(x, state)
            new_states.append(s)

        x = self.norm(x)
        logits = self.head(x)
        return logits, new_states

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
