"""Long ES training run to find convergence limit.

Runs ES for 3000 steps with larger population to see how close
we can get to backprop's loss of 1.62. The 500-step run hit 3.18
and was still dropping — this should reveal the true ceiling.

Run: python experiments/train_es_long.py
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ngai.data.text_dataset import CharDataset
from ngai.evolve.es_trainer import ESTrainer
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
ES_SIGMA = 0.1
ES_LR = 0.01
ES_MOMENTUM = 0.9


BACKPROP_REFERENCE_LOSS = 1.62


def main() -> None:
    """Run long ES training to find convergence limit."""
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
        f"ES: pop={ES_POP} pairs ({ES_POP * 2} evals), "
        f"sigma={ES_SIGMA}, lr={ES_LR}, momentum={ES_MOMENTUM}"
    )
    print(f"Training for {STEPS} steps (no backprop)...\n")

    trainer = ESTrainer(
        model, pop_size=ES_POP, sigma=ES_SIGMA,
        lr=ES_LR, momentum=ES_MOMENTUM, device=DEVICE,
    )
    losses = trainer.train(loader, STEPS)

    final = sum(losses[-100:]) / min(100, len(losses))
    print(f"\nFinal avg loss (last 100): {final:.4f}")
    print(f"Best loss:                 {trainer.best_loss:.4f}")
    print(f"Final sigma:               {trainer.sigma:.4f}")

    print(f"\nReference: backprop achieves ~{BACKPROP_REFERENCE_LOSS} on same data/model")
    gap = final / BACKPROP_REFERENCE_LOSS
    print(f"Current gap: {gap:.1f}x backprop loss")


if __name__ == "__main__":
    main()
