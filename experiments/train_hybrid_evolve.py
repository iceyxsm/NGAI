"""Train NGAI using the hybrid evolutionary trainer.

Compares the new hybrid approach (goodness-guided per-layer mutation)
against the baseline evolutionary trainer. Both are gradient-free.

The hybrid trainer should converge faster because:
1. Per-layer mutation focuses search on weak layers
2. Forward-Forward goodness provides local quality signals
3. Larger population (64 vs 32) explores more of the space

Run: python experiments/train_hybrid_evolve.py
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ngai.data.text_dataset import CharDataset
from ngai.evolve.hybrid_trainer import HybridEvolveTrainer
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

HYBRID_POP = 64
HYBRID_RATE = 0.0005
HYBRID_GOODNESS_SCALE = 2.0
HYBRID_GOODNESS_INTERVAL = 10

BASELINE_POP = 32
BASELINE_RATE = 0.0001


def make_loader(dataset: CharDataset) -> DataLoader:
    """Create a DataLoader from a CharDataset."""
    return DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
    )


def run_baseline(dataset: CharDataset, vocab: int) -> list[float]:
    """Run baseline evolutionary trainer."""
    print("=" * 60)
    print("BASELINE: Standard Evolutionary Trainer")
    print(f"  Pop: {BASELINE_POP}, Rate: {BASELINE_RATE}")
    print("=" * 60)

    set_seed(42)
    model = NGAILanguageModel(vocab, DIM, N_LAYERS).to(DEVICE)
    params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {params:,} params")

    trainer = EvolveTrainer(
        model, pop_size=BASELINE_POP,
        mutation_rate=BASELINE_RATE, device=DEVICE,
    )
    loader = make_loader(dataset)
    losses = trainer.train(loader, STEPS)

    final = sum(losses[-50:]) / min(50, len(losses))
    print(f"\n  Baseline final loss: {final:.4f}")
    print(f"  Baseline best loss:  {trainer.best_loss:.4f}")
    return losses


def run_hybrid(dataset: CharDataset, vocab: int) -> list[float]:
    """Run hybrid evolutionary trainer with guided mutation."""
    print("\n" + "=" * 60)
    print("HYBRID: Goodness-Guided Evolutionary Trainer")
    print(
        f"  Pop: {HYBRID_POP}, Rate: {HYBRID_RATE}, "
        f"Goodness scale: {HYBRID_GOODNESS_SCALE}"
    )
    print("=" * 60)

    set_seed(42)
    model = NGAILanguageModel(vocab, DIM, N_LAYERS).to(DEVICE)
    params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {params:,} params")

    trainer = HybridEvolveTrainer(
        model, pop_size=HYBRID_POP,
        mutation_rate=HYBRID_RATE,
        goodness_scale=HYBRID_GOODNESS_SCALE,
        goodness_interval=HYBRID_GOODNESS_INTERVAL,
        device=DEVICE,
    )
    loader = make_loader(dataset)
    losses = trainer.train(loader, STEPS)

    final = sum(losses[-50:]) / min(50, len(losses))
    print(f"\n  Hybrid final loss: {final:.4f}")
    print(f"  Hybrid best loss:  {trainer.best_loss:.4f}")

    diag = trainer.get_layer_diagnostics()
    print("\n  Per-layer mutation rates:")
    for name, rate in diag.items():
        print(f"    {name}: {rate:.6f}")

    return losses


def main() -> None:
    """Run hybrid vs baseline comparison."""
    print(f"Device: {DEVICE}")
    print("Loading TinyShakespeare...")

    dataset = CharDataset.from_file(DATA_PATH, SEQ_LEN)
    dataset.data = dataset.data[:MAX_CHARS]
    vocab = dataset.vocab_size
    print(f"Vocab: {vocab}, Data: {len(dataset):,} seqs\n")

    baseline_losses = run_baseline(dataset, vocab)
    hybrid_losses = run_hybrid(dataset, vocab)

    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)

    window = min(50, len(baseline_losses), len(hybrid_losses))
    bl_final = sum(baseline_losses[-window:]) / window
    hy_final = sum(hybrid_losses[-window:]) / window
    improvement = (bl_final - hy_final) / bl_final * 100

    print(f"  Baseline final avg loss: {bl_final:.4f}")
    print(f"  Hybrid final avg loss:   {hy_final:.4f}")
    print(f"  Improvement:             {improvement:+.1f}%")


if __name__ == "__main__":
    main()
