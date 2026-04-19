"""ES training at 10M scale — gradient-free on a real model.

Uses layer-wise ES: perturbs one layer at a time instead of all 8M
params simultaneously. This reduces the search dimensionality from
8M to ~500K per step, giving much cleaner gradient estimates.

Backprop baseline on this config: ~0.95 loss at 5000 steps.
Target: ES loss under 2.0 would be a strong result at this scale.

Run on GPU: python experiments/train_es_10m.py
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ngai.data.text_dataset import CharDataset
from ngai.evolve.layerwise_es import LayerwiseESTrainer
from ngai.models.ngai_lm import NGAILanguageModel
from ngai.utils.seed import set_seed

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_PATH = Path("data/tinyshakespeare.txt")
DIM = 256
N_LAYERS = 8
SEQ_LEN = 256
BATCH_SIZE = 32
STEPS = 3000

POP_SIZE = 32
SIGMA = 0.05
LR = 0.005
MOMENTUM = 0.9
WEIGHT_DECAY = 0.001

BACKPROP_REFERENCE_LOSS = 0.95


def main() -> None:
    """Run layer-wise ES at 10M scale."""
    set_seed(42)
    print(f"Device: {DEVICE}")
    print("Loading TinyShakespeare...")

    dataset = CharDataset.from_file(DATA_PATH, SEQ_LEN)
    vocab = dataset.vocab_size
    loader: DataLoader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
    )
    print(f"Vocab: {vocab}, Data: {len(dataset):,} seqs\n")

    model = NGAILanguageModel(vocab, DIM, N_LAYERS).to(DEVICE)
    params = sum(p.numel() for p in model.parameters())

    print("=" * 55)
    print("  Layer-wise ES (no backprop)")
    print(f"  Model: {params:,} params, dim={DIM}, layers={N_LAYERS}")
    print(f"  Pop={POP_SIZE}, sigma={SIGMA}, lr={LR}")
    print("=" * 55)

    trainer = LayerwiseESTrainer(
        model, pop_size=POP_SIZE, sigma=SIGMA,
        lr=LR, momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY, device=DEVICE,
    )
    losses = trainer.train(loader, STEPS)

    window = min(100, len(losses))
    final = sum(losses[-window:]) / window

    print("\n" + "=" * 55)
    print("  10M SCALE LAYER-WISE ES RESULTS")
    print("=" * 55)
    print(f"  Final avg loss:      {final:.4f}")
    print(f"  Best loss:           {trainer.best_loss:.4f}")
    print(f"  Backprop reference:  ~{BACKPROP_REFERENCE_LOSS}")
    print(f"  Gap:                 {final / BACKPROP_REFERENCE_LOSS:.1f}x")
    print("=" * 55)


if __name__ == "__main__":
    main()
