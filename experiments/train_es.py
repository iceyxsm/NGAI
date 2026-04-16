"""Train NGAI using Natural Evolution Strategies.

3-way comparison: baseline random mutation vs goodness-guided vs ES.
ES uses antithetic sampling + fitness shaping + momentum — a proper
gradient estimator without backpropagation.

Run: python experiments/train_es.py
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ngai.data.text_dataset import CharDataset
from ngai.evolve.es_trainer import ESTrainer
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

BASELINE_POP = 32
BASELINE_RATE = 0.0001

ES_POP = 16
ES_SIGMA = 0.1
ES_LR = 0.01
ES_MOMENTUM = 0.9


def make_loader(dataset: CharDataset) -> DataLoader:
    """Create a DataLoader from a CharDataset."""
    return DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
    )


def run_baseline(dataset: CharDataset, vocab: int) -> list[float]:
    """Run baseline evolutionary trainer."""
    print("=" * 60)
    print("BASELINE: Random Mutation + Selection")
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
    losses = trainer.train(make_loader(dataset), STEPS)

    final = sum(losses[-50:]) / min(50, len(losses))
    print(f"\n  Baseline final loss: {final:.4f}")
    print(f"  Baseline best loss:  {trainer.best_loss:.4f}")
    return losses


def run_es(dataset: CharDataset, vocab: int) -> list[float]:
    """Run Natural Evolution Strategy trainer."""
    print("\n" + "=" * 60)
    print("ES: Antithetic Sampling + Fitness Shaping + Momentum")
    print(
        f"  Pop: {ES_POP} pairs ({ES_POP * 2} evals), "
        f"Sigma: {ES_SIGMA}, LR: {ES_LR}, Mom: {ES_MOMENTUM}"
    )
    print("=" * 60)

    set_seed(42)
    model = NGAILanguageModel(vocab, DIM, N_LAYERS).to(DEVICE)
    params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {params:,} params")

    trainer = ESTrainer(
        model, pop_size=ES_POP, sigma=ES_SIGMA,
        lr=ES_LR, momentum=ES_MOMENTUM, device=DEVICE,
    )
    losses = trainer.train(make_loader(dataset), STEPS)

    final = sum(losses[-50:]) / min(50, len(losses))
    print(f"\n  ES final loss: {final:.4f}")
    print(f"  ES best loss:  {trainer.best_loss:.4f}")
    return losses


def main() -> None:
    """Run baseline vs ES comparison."""
    print(f"Device: {DEVICE}")
    print("Loading TinyShakespeare...")

    dataset = CharDataset.from_file(DATA_PATH, SEQ_LEN)
    dataset.data = dataset.data[:MAX_CHARS]
    vocab = dataset.vocab_size
    print(f"Vocab: {vocab}, Data: {len(dataset):,} seqs\n")

    baseline_losses = run_baseline(dataset, vocab)
    es_losses = run_es(dataset, vocab)

    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)

    window = min(50, len(baseline_losses), len(es_losses))
    bl_final = sum(baseline_losses[-window:]) / window
    es_final = sum(es_losses[-window:]) / window
    improvement = (bl_final - es_final) / bl_final * 100

    print(f"  Baseline final avg loss: {bl_final:.4f}")
    print(f"  ES final avg loss:       {es_final:.4f}")
    print(f"  Improvement:             {improvement:+.1f}%")


if __name__ == "__main__":
    main()
