"""Compare novel routing strategies on TinyShakespeare.

Trains 4 MoE variants and compares loss:
1. Standard MoE (baseline)
2. State-Aware Router
3. Adaptive Router
4. Standard MoE with CrossLayer Expert Pool

Run: uv run python experiments/compare_novel_routing.py
"""

import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ngai.core.novel_routing import AdaptiveRouter, StateAwareRouter
from ngai.data.text_dataset import CharDataset
from ngai.models.ngai_moe_lm import NGAIMoELanguageModel
from ngai.utils.seed import set_seed

DATA_PATH = Path("data/tinyshakespeare.txt")
DIM = 128
N_LAYERS = 4
SEQ_LEN = 64
BATCH_SIZE = 64
LR = 5e-4
STEPS = 300
LOG_EVERY = 100
MAX_CHARS = 100_000

N_SHARED = 1
N_ROUTED = 8
TOP_K = 2
MOE_EXPAND = 2


def make_model(vocab_size: int, router_type: str) -> NGAIMoELanguageModel:
    """Create an MoE model and swap in the specified router type."""
    model = NGAIMoELanguageModel(
        vocab_size, DIM, N_LAYERS, N_SHARED, N_ROUTED, TOP_K, MOE_EXPAND
    )
    if router_type == "standard":
        pass  # already uses ExpertRouter
    elif router_type == "state_aware":
        for block in model.blocks:
            block.channel_mixer.router = StateAwareRouter(DIM, N_ROUTED, TOP_K)
    elif router_type == "adaptive":
        for block in model.blocks:
            block.channel_mixer.router = AdaptiveRouter(DIM, N_ROUTED, TOP_K)
    return model


def train_model(
    name: str,
    model: NGAIMoELanguageModel,
    loader: DataLoader,
    router_type: str,
) -> dict:
    """Train one model variant and return results."""
    print(f"\n  Training: {name}")
    print(f"  Params: {model.count_parameters():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    model.train()

    loader_iter = iter(loader)
    losses = []
    t0 = time.perf_counter()

    for step in range(STEPS):
        try:
            x, y = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            x, y = next(loader_iter)

        logits, _, balance_loss = model(x)
        ce_loss = criterion(logits.view(-1, model.vocab_size), y.view(-1))
        loss = ce_loss + balance_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        losses.append(ce_loss.item())
        if (step + 1) % LOG_EVERY == 0:
            avg = sum(losses[-LOG_EVERY:]) / LOG_EVERY
            print(f"    step {step+1:>4} | loss {avg:.4f}")

    elapsed = time.perf_counter() - t0
    final_loss = sum(losses[-50:]) / 50 if len(losses) >= 50 else sum(losses) / len(losses)

    return {
        "name": name,
        "router": router_type,
        "params": model.count_parameters(),
        "final_loss": final_loss,
        "time": elapsed,
        "steps_per_sec": STEPS / elapsed,
    }


def main() -> None:
    """Run the novel routing comparison experiment."""
    set_seed(42)
    print("Loading TinyShakespeare...")
    dataset = CharDataset.from_file(DATA_PATH, SEQ_LEN)
    dataset.data = dataset.data[:MAX_CHARS]
    loader: DataLoader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
    )
    vocab = dataset.vocab_size
    print(f"Vocab: {vocab}, Data: {len(dataset):,} seqs")

    variants = [
        ("Standard MoE", "standard"),
        ("State-Aware Router", "state_aware"),
        ("Adaptive Router", "adaptive"),
    ]

    results = []
    for name, router_type in variants:
        set_seed(42)
        model = make_model(vocab, router_type)
        result = train_model(name, model, loader, router_type)
        results.append(result)

    # Print comparison
    print(f"\n{'='*60}")
    print("  NOVEL ROUTING COMPARISON")
    print(f"{'='*60}")
    print(f"  {'Model':<25} {'Params':>10} {'Loss':>8} {'Step/s':>8}")
    print(f"  {'-'*25} {'-'*10} {'-'*8} {'-'*8}")
    for r in results:
        print(
            f"  {r['name']:<25} {r['params']:>10,} "
            f"{r['final_loss']:>8.4f} {r['steps_per_sec']:>8.2f}"
        )
    print(f"{'='*60}")

    # Find best
    best = min(results, key=lambda r: r["final_loss"])
    print(f"\n  Best: {best['name']} (loss {best['final_loss']:.4f})")


if __name__ == "__main__":
    main()
