"""Train NGAI using gradient-free evolutionary training.

The hybrid approach: ternary mutations + population selection.
No backpropagation. No gradients. Pure forward passes.

Run: python experiments/train_evolve.py
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ngai.data.text_dataset import CharDataset
from ngai.evolve.trainer import EvolveTrainer
from ngai.models.ngai_lm import NGAILanguageModel
from ngai.utils.seed import set_seed

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_PATH = Path("data/tinyshakespeare.txt")
DIM = 128
N_LAYERS = 4
SEQ_LEN = 64
BATCH_SIZE = 32
MAX_CHARS = 50_000
STEPS = 500
POP_SIZE = 16
MUTATION_RATE = 0.005


def main() -> None:
    """Run evolutionary training experiment."""
    set_seed(42)
    print(f"Device: {DEVICE}")
    print("Loading TinyShakespeare...")

    dataset = CharDataset.from_file(DATA_PATH, SEQ_LEN)
    dataset.data = dataset.data[:MAX_CHARS]
    loader: DataLoader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
    )
    vocab = dataset.vocab_size
    print(f"Vocab: {vocab}, Data: {len(dataset):,} seqs")

    model = NGAILanguageModel(vocab, DIM, N_LAYERS).to(DEVICE)
    params = sum(p.numel() for p in model.parameters())
    print(f"Model: {params:,} params")
    print(f"Pop: {POP_SIZE}, Mutation: {MUTATION_RATE}")
    print(f"Training for {STEPS} steps (no backprop)...\n")

    trainer = EvolveTrainer(
        model, pop_size=POP_SIZE,
        mutation_rate=MUTATION_RATE, device=DEVICE,
    )
    losses = trainer.train(loader, STEPS)

    final = sum(losses[-50:]) / 50 if len(losses) >= 50 else sum(losses) / len(losses)
    print(f"\nFinal loss: {final:.4f}")
    print(f"Best loss:  {trainer.best_loss:.4f}")


if __name__ == "__main__":
    main()
