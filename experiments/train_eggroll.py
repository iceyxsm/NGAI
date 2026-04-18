"""EGGROLL ES training — low-rank perturbations for GPU efficiency.

Same algorithm as serial ES but with structured low-rank noise.
Reduces noise generation 500x and gradient update cost proportionally.

Run: python experiments/train_eggroll.py
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ngai.data.text_dataset import CharDataset
from ngai.evolve.eggroll_trainer import EggrollTrainer
from ngai.models.ngai_lm import NGAILanguageModel
from ngai.utils.seed import set_seed

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_PATH = Path("data/tinyshakespeare.txt")
DIM = 128
N_LAYERS = 4
SEQ_LEN = 64
BATCH_SIZE = 32
MAX_CHARS = 100_000
STEPS = 3000

POP_SIZE = 32
RANK = 1
SIGMA = 0.1
LR = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 0.001

BACKPROP_REFERENCE_LOSS = 1.62


def main() -> None:
    """Run EGGROLL training."""
    set_seed(42)
    print(f"Device: {DEVICE}")
    print("Loading TinyShakespeare...")

    dataset = CharDataset.from_file(DATA_PATH, SEQ_LEN)
    dataset.data = dataset.data[:MAX_CHARS]
    vocab = dataset.vocab_size
    loader: DataLoader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
    )
    print(f"Vocab: {vocab}, Data: {len(dataset):,} seqs")

    model = NGAILanguageModel(vocab, DIM, N_LAYERS).to(DEVICE)
    params = sum(p.numel() for p in model.parameters())
    print(f"Model: {params:,} params")
    print(
        f"EGGROLL: pop={POP_SIZE} pairs, rank={RANK}, "
        f"sigma={SIGMA}, lr={LR}, momentum={MOMENTUM}"
    )
    print(f"Training for {STEPS} steps (no backprop)...\n")

    trainer = EggrollTrainer(
        model, pop_size=POP_SIZE, rank=RANK, sigma=SIGMA,
        lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY,
        device=DEVICE,
    )
    losses = trainer.train(loader, STEPS)

    final = sum(losses[-100:]) / min(100, len(losses))
    print(f"\nFinal avg loss (last 100): {final:.4f}")
    print(f"Best loss:                 {trainer.best_loss:.4f}")

    gap = final / BACKPROP_REFERENCE_LOSS
    print(f"\nReference: backprop ~{BACKPROP_REFERENCE_LOSS}")
    print(f"Current gap: {gap:.1f}x backprop loss")


if __name__ == "__main__":
    main()
