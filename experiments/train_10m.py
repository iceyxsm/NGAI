"""Train a 10M parameter NGAI model on full TinyShakespeare.

Scales up: dim=512, 8 layers, full dataset, 5000 steps.
Compares dense 10M vs MoE 10M (total) with ~2M active.

Run on GPU: python experiments/train_10m.py
"""

import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ngai.data.text_dataset import CharDataset
from ngai.models.ngai_lm import NGAILanguageModel
from ngai.models.ngai_moe_lm import NGAIMoELanguageModel
from ngai.utils.seed import set_seed

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_PATH = Path("data/tinyshakespeare.txt")
SEQ_LEN = 128
BATCH_SIZE = 64
LR = 3e-4
STEPS = 5000
LOG_EVERY = 500

# Dense 10M: dim=512, 8 layers
DENSE_DIM = 512
DENSE_LAYERS = 8

# MoE ~10M total: dim=256, 8 layers, 16 routed experts
MOE_DIM = 256
MOE_LAYERS = 8
N_SHARED = 1
N_ROUTED = 16
TOP_K = 2
MOE_EXPAND = 2


def train_model(
    name: str,
    model: nn.Module,
    loader: DataLoader,
    is_moe: bool = False,
) -> dict:
    """Train a model and return results."""
    params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"  Params: {params:,} | Device: {DEVICE}")
    print(f"{'='*55}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    model.train()

    loader_iter = iter(loader)
    losses: list[float] = []
    t0 = time.perf_counter()

    for step in range(STEPS):
        try:
            x, y = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            x, y = next(loader_iter)

        x, y = x.to(DEVICE), y.to(DEVICE)

        if is_moe:
            logits, _, bl = model(x)
            vocab = model.vocab_size
            ce = criterion(logits.view(-1, vocab), y.view(-1))
            loss = ce + bl
        else:
            logits, _ = model(x)
            vocab = model.vocab_size
            ce = criterion(logits.view(-1, vocab), y.view(-1))
            loss = ce

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(ce.item())

        if (step + 1) % LOG_EVERY == 0:
            avg = sum(losses[-LOG_EVERY:]) / LOG_EVERY
            elapsed = time.perf_counter() - t0
            sps = (step + 1) / elapsed
            print(f"  step {step+1:>5} | loss {avg:.4f} | {sps:.1f} steps/s")

    elapsed = time.perf_counter() - t0
    final = sum(losses[-100:]) / 100
    return {"name": name, "params": params, "loss": final, "time": elapsed}


@torch.no_grad()
def generate(model: nn.Module, dataset: CharDataset, prompt: str, is_moe: bool = False) -> str:
    """Generate text sample."""
    model.eval()
    idx = [dataset.char_to_idx.get(c, 0) for c in prompt]
    tokens = torch.tensor([idx], dtype=torch.long, device=DEVICE)
    if is_moe:
        logits, states, _ = model(tokens)
    else:
        logits, states = model(tokens)
    out = list(prompt)
    for _ in range(300):
        probs = torch.softmax(logits[:, -1, :] / 0.8, dim=-1)
        tok = torch.multinomial(probs, 1)
        out.append(dataset.idx_to_char[tok.item()])
        if is_moe:
            logits, states, _ = model(tok, states)
        else:
            logits, states = model(tok, states)
    return "".join(out)


def main() -> None:
    """Train 10M dense and MoE models."""
    set_seed(42)
    print(f"Device: {DEVICE}")
    print("Loading TinyShakespeare (full)...")
    dataset = CharDataset.from_file(DATA_PATH, SEQ_LEN)
    loader: DataLoader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
    )
    vocab = dataset.vocab_size
    print(f"Vocab: {vocab}, Sequences: {len(dataset):,}\n")

    # Dense 10M
    set_seed(42)
    dense = NGAILanguageModel(vocab, DENSE_DIM, DENSE_LAYERS).to(DEVICE)
    d = train_model("Dense 10M", dense, loader)

    # MoE ~10M total
    set_seed(42)
    moe = NGAIMoELanguageModel(
        vocab, MOE_DIM, MOE_LAYERS, N_SHARED, N_ROUTED, TOP_K, MOE_EXPAND
    ).to(DEVICE)
    m = train_model("MoE 10M", moe, loader, is_moe=True)

    # Results
    print(f"\n{'='*55}")
    print("  10M SCALE RESULTS")
    print(f"{'='*55}")
    print(f"  {'Model':<20} {'Params':>12} {'Loss':>8} {'Time':>8}")
    print(f"  {'-'*20} {'-'*12} {'-'*8} {'-'*8}")
    for r in [d, m]:
        print(f"  {r['name']:<20} {r['params']:>12,} {r['loss']:>8.4f} {r['time']:>7.1f}s")
    print(f"{'='*55}")

    # Samples
    print("\n  Dense sample:")
    print(f"  {generate(dense, dataset, 'ROMEO:\\n')[:300]}")
    print("\n  MoE sample:")
    print(f"  {generate(moe, dataset, 'ROMEO:\\n', is_moe=True)[:300]}")


if __name__ == "__main__":
    main()
