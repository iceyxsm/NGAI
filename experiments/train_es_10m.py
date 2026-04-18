"""ES training at 10M scale — gradient-free on a real model.

Runs both full ES and EGGROLL on a 10M param model (dim=256, 8 layers)
to see if the 1.5x backprop ratio holds at scale.

Backprop baseline on this config: ~1.2-1.4 loss at 5000 steps.
Target: ES loss under 2.5 would be groundbreaking at this scale.

Run on GPU: python experiments/train_es_10m.py
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ngai.data.text_dataset import CharDataset
from ngai.evolve.eggroll_trainer import EggrollTrainer
from ngai.evolve.es_trainer import ESTrainer
from ngai.models.ngai_lm import NGAILanguageModel
from ngai.utils.seed import set_seed

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_PATH = Path("data/tinyshakespeare.txt")
DIM = 256
N_LAYERS = 8 
SEQ_LEN = 256  # 2x longer sequences
BATCH_SIZE = 32  # Reduced for VRAM at 8M scale
STEPS = 3000

POP_SIZE = 16  # Reduced for VRAM (8M model × 32 variants = 1GB)
SIGMA = 0.05
LR = 0.005
MOMENTUM = 0.9
WEIGHT_DECAY = 0.001
EGGROLL_RANK = 4

BACKPROP_REFERENCE_LOSS = 1.3


def run_eggroll(model: NGAILanguageModel, loader: DataLoader) -> list[float]:
    """Run EGGROLL ES training."""
    params = sum(p.numel() for p in model.parameters())
    print("=" * 55)
    print("  EGGROLL ES (low-rank, no backprop)")
    print(f"  Model: {params:,} params, dim={DIM}, layers={N_LAYERS}")
    print(
        f"  Pop={POP_SIZE}, rank={EGGROLL_RANK}, "
        f"sigma={SIGMA}, lr={LR}"
    )
    print("=" * 55)

    trainer = EggrollTrainer(
        model, pop_size=POP_SIZE, rank=EGGROLL_RANK,
        sigma=SIGMA, lr=LR, momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY, device=DEVICE,
    )
    return trainer.train(loader, STEPS)


def run_full_es(model: NGAILanguageModel, loader: DataLoader) -> list[float]:
    """Run full ES training."""
    params = sum(p.numel() for p in model.parameters())
    print("\n" + "=" * 55)
    print("  Full ES (antithetic + momentum, no backprop)")
    print(f"  Model: {params:,} params, dim={DIM}, layers={N_LAYERS}")
    print(f"  Pop={POP_SIZE}, sigma={SIGMA}, lr={LR}")
    print("=" * 55)

    trainer = ESTrainer(
        model, pop_size=POP_SIZE, sigma=SIGMA,
        lr=LR, lr_min_ratio=1.0, momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY, eval_batches=1,
        device=DEVICE,
    )
    return trainer.train(loader, STEPS)


def main() -> None:
    """Run full ES at 10M scale."""
    set_seed(42)
    print(f"Device: {DEVICE}")
    print("Loading TinyShakespeare...")

    dataset = CharDataset.from_file(DATA_PATH, SEQ_LEN)
    vocab = dataset.vocab_size
    loader: DataLoader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
    )
    print(f"Vocab: {vocab}, Data: {len(dataset):,} seqs\n")

    set_seed(42)
    model_es = NGAILanguageModel(vocab, DIM, N_LAYERS).to(DEVICE)
    es_losses = run_full_es(model_es, loader)

    es_window = min(100, len(es_losses))
    es_final = sum(es_losses[-es_window:]) / es_window

    print("\n" + "=" * 55)
    print("  10M SCALE ES RESULTS")
    print("=" * 55)
    print(f"  Full ES final loss:  {es_final:.4f}")
    print(f"  Backprop reference:  ~{BACKPROP_REFERENCE_LOSS}")
    print(f"  Full ES gap:         {es_final / BACKPROP_REFERENCE_LOSS:.1f}x")
    print("=" * 55)


if __name__ == "__main__":
    main()
