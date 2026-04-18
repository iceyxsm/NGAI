"""Fast ES training with activation-space perturbation.

One forward pass evaluates all 64 variants simultaneously.
Expected 15-20x speedup over serial ES.

Run: python experiments/train_fast_es.py
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ngai.data.text_dataset import CharDataset
from ngai.evolve.fast_es import ActivationPerturbES
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

ES_POP = 32
ES_SIGMA = 0.02
ES_LR = 0.01
ES_MOMENTUM = 0.9
ES_WEIGHT_DECAY = 0.001

BACKPROP_REFERENCE_LOSS = 1.62


def main() -> None:
    """Run fast ES training."""
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
        f"Fast ES: pop={ES_POP} pairs, sigma={ES_SIGMA}, "
        f"lr={ES_LR}, momentum={ES_MOMENTUM}"
    )
    print(f"Training for {STEPS} steps (no backprop)...\n")

    trainer = ActivationPerturbES(
        model, pop_size=ES_POP, sigma=ES_SIGMA,
        lr=ES_LR, momentum=ES_MOMENTUM,
        weight_decay=ES_WEIGHT_DECAY, device=DEVICE,
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
