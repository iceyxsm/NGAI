"""Simple character-level text dataset for NGAI training experiments.

Loads a text file, builds a character vocabulary, and produces
fixed-length sequences for causal language modeling.
"""

from pathlib import Path

import torch
from torch.utils.data import Dataset


class CharDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Character-level dataset for language modeling.

    Tokenizes text at the character level and produces
    (input, target) pairs where target is input shifted by 1.

    Args:
        text: Raw text string to tokenize.
        seq_len: Length of each training sequence.
    """

    def __init__(self, text: str, seq_len: int) -> None:
        self.seq_len = seq_len
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.char_to_idx = {c: i for i, c in enumerate(chars)}
        self.idx_to_char = {i: c for i, c in enumerate(chars)}
        self.data = torch.tensor(
            [self.char_to_idx[c] for c in text], dtype=torch.long
        )

    def __len__(self) -> int:
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a (input, target) pair."""
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + 1 : idx + self.seq_len + 1]
        return x, y

    def decode(self, indices: torch.Tensor) -> str:
        """Convert token indices back to text."""
        return "".join(self.idx_to_char[i.item()] for i in indices)

    @classmethod
    def from_file(cls, path: str | Path, seq_len: int) -> "CharDataset":
        """Load dataset from a text file.

        Args:
            path: Path to the text file.
            seq_len: Sequence length for training.
        """
        text = Path(path).read_text(encoding="utf-8")
        return cls(text, seq_len)
