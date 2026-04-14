"""Train a tiny NGAI language model on TinyShakespeare.

Proof-of-concept: can the ternary MatMul-free architecture learn language?

Run: uv run python experiments/train_tiny_lm.py
"""

import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ngai.data.text_dataset import CharDataset
from ngai.models.ngai_lm import NGAILanguageModel
from ngai.utils.seed import set_seed

DATA_PATH = Path("data/tinyshakespeare.txt")
DIM = 128
N_LAYERS = 4
SEQ_LEN = 64
BATCH_SIZE = 64
LR = 5e-4
EPOCHS = 1
LOG_EVERY = 50
SAMPLE_LEN = 200
MAX_CHARS = 100_000


def train_epoch(
    model: NGAILanguageModel,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    epoch: int,
) -> float:
    """Train for one epoch, return average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for i, (x, y) in enumerate(loader):
        logits, _ = model(x)
        loss = criterion(logits.view(-1, model.vocab_size), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        if (i + 1) % LOG_EVERY == 0:
            avg = total_loss / n_batches
            print(f"  epoch {epoch+1} | step {i+1:>5} | loss {avg:.4f}")

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def generate(
    model: NGAILanguageModel,
    dataset: CharDataset,
    prompt: str,
    max_tokens: int = SAMPLE_LEN,
) -> str:
    """Generate text from a prompt."""
    model.eval()
    indices = [dataset.char_to_idx.get(c, 0) for c in prompt]
    tokens = torch.tensor([indices], dtype=torch.long)
    states: list[torch.Tensor] | None = None

    logits, states = model(tokens, states)

    generated = list(prompt)
    for _ in range(max_tokens):
        next_logit = logits[:, -1, :]
        probs = torch.softmax(next_logit / 0.8, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated.append(dataset.idx_to_char[next_token.item()])
        logits, states = model(next_token, states)

    return "".join(generated)


def main() -> None:
    """Train and evaluate the tiny NGAI language model."""
    set_seed(42)
    print("Loading TinyShakespeare...")
    dataset = CharDataset.from_file(DATA_PATH, SEQ_LEN)
    if MAX_CHARS:
        dataset.data = dataset.data[:MAX_CHARS]
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
        num_workers=4, persistent_workers=True,
    )

    print(f"Vocab size: {dataset.vocab_size}")
    print(f"Dataset size: {len(dataset):,} sequences")

    model = NGAILanguageModel(
        vocab_size=dataset.vocab_size, dim=DIM, n_layers=N_LAYERS
    )
    print(f"Model params: {model.count_parameters():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    print(f"\nTraining for {EPOCHS} epochs...")
    start = time.perf_counter()

    for epoch in range(EPOCHS):
        avg_loss = train_epoch(model, loader, optimizer, criterion, epoch)
        elapsed = time.perf_counter() - start
        print(f"Epoch {epoch+1}/{EPOCHS} | loss: {avg_loss:.4f} | time: {elapsed:.1f}s")

    print("\nGenerating sample...")
    sample = generate(model, dataset, "ROMEO:\n")
    print(f"\n{'='*50}")
    print(sample)
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
