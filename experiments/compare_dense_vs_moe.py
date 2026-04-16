"""Compare Dense vs MoE NGAI models on TinyShakespeare.

Trains both models on the same data and compares:
- Final loss
- Total params vs active params
- Training speed (steps/sec)
- Generated text quality

Run: uv run python experiments/compare_dense_vs_moe.py
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
DIM = 128
N_LAYERS = 4
SEQ_LEN = 64
BATCH_SIZE = 64
LR = 5e-4
STEPS = 500
LOG_EVERY = 100
MAX_CHARS = 100_000
SAMPLE_LEN = 150

# MoE config
N_SHARED = 1
N_ROUTED = 8
TOP_K = 2
MOE_EXPAND = 2


def train_steps(
    model: nn.Module,
    loader_iter: object,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    steps: int,
    is_moe: bool = False,
) -> list[float]:
    """Train for a fixed number of steps, return loss history."""
    model.train()
    losses: list[float] = []

    for step in range(steps):
        try:
            x, y = next(loader_iter)
        except StopIteration:
            return losses

        x, y = x.to(DEVICE), y.to(DEVICE)

        if is_moe:
            logits, _, balance_loss = model(x)
            ce_loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            loss = ce_loss + balance_loss
        else:
            logits, _ = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        losses.append(loss.item())
        if (step + 1) % LOG_EVERY == 0:
            avg = sum(losses[-LOG_EVERY:]) / LOG_EVERY
            print(f"    step {step+1:>4} | loss {avg:.4f}")

    return losses


@torch.no_grad()
def generate(model: nn.Module, dataset: CharDataset, prompt: str) -> str:
    """Generate text from a model."""
    model.eval()
    indices = [dataset.char_to_idx.get(c, 0) for c in prompt]
    tokens = torch.tensor([indices], dtype=torch.long, device=DEVICE)
    is_moe = isinstance(model, NGAIMoELanguageModel)

    if is_moe:
        logits, states, _ = model(tokens)
    else:
        logits, states = model(tokens)

    generated = list(prompt)
    for _ in range(SAMPLE_LEN):
        next_logit = logits[:, -1, :]
        probs = torch.softmax(next_logit / 0.8, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated.append(dataset.idx_to_char[next_token.item()])
        if is_moe:
            logits, states, _ = model(next_token, states)
        else:
            logits, states = model(next_token, states)
    return "".join(generated)


def main() -> None:
    """Train and compare dense vs MoE models."""
    set_seed(42)
    print("Loading TinyShakespeare...")
    dataset = CharDataset.from_file(DATA_PATH, SEQ_LEN)
    dataset.data = dataset.data[:MAX_CHARS]
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
    )
    vocab = dataset.vocab_size
    print(f"Device: {DEVICE}")
    print(f"Vocab: {vocab}, Data: {len(dataset):,} seqs\n")

    # --- Dense model ---
    print("=" * 55)
    print("  DENSE MODEL (NGAILanguageModel)")
    print("=" * 55)
    set_seed(42)
    dense = NGAILanguageModel(vocab, DIM, N_LAYERS).to(DEVICE)
    dense_params = dense.count_parameters()
    print(f"  Total params: {dense_params:,}")
    opt_d = torch.optim.AdamW(dense.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()
    t0 = time.perf_counter()
    d_losses = train_steps(dense, iter(loader), opt_d, crit, STEPS)
    d_time = time.perf_counter() - t0
    d_sample = generate(dense, dataset, "ROMEO:\n")

    # --- MoE model ---
    print(f"\n{'=' * 55}")
    print("  MOE MODEL (NGAIMoELanguageModel)")
    print("=" * 55)
    set_seed(42)
    moe = NGAIMoELanguageModel(
        vocab, DIM, N_LAYERS, N_SHARED, N_ROUTED, TOP_K, MOE_EXPAND
    ).to(DEVICE)
    moe_total = moe.count_parameters()
    moe_active = moe.count_active_parameters()
    print(f"  Total params: {moe_total:,}")
    print(f"  Active/token: {moe_active:,}")
    print(f"  Sparsity:     {1 - moe_active/moe_total:.1%}")
    opt_m = torch.optim.AdamW(moe.parameters(), lr=LR)
    t0 = time.perf_counter()
    m_losses = train_steps(moe, iter(loader), opt_m, crit, STEPS, is_moe=True)
    m_time = time.perf_counter() - t0
    m_sample = generate(moe, dataset, "ROMEO:\n")

    # --- Results ---
    d_final = sum(d_losses[-50:]) / 50 if d_losses else 0
    m_final = sum(m_losses[-50:]) / 50 if m_losses else 0
    print(f"\n{'=' * 55}")
    print("  COMPARISON RESULTS")
    print(f"{'=' * 55}")
    print(f"  {'':28} {'Dense':>10} {'MoE':>10}")
    print(f"  {'-'*28} {'-'*10} {'-'*10}")
    print(f"  {'Total params':28} {dense_params:>10,} {moe_total:>10,}")
    print(f"  {'Active params/token':28} {dense_params:>10,} {moe_active:>10,}")
    print(f"  {'Final loss (avg last 50)':28} {d_final:>10.4f} {m_final:>10.4f}")
    print(f"  {'Training time (s)':28} {d_time:>10.1f} {m_time:>10.1f}")
    print(f"  {'Steps/sec':28} {STEPS/d_time:>10.1f} {STEPS/m_time:>10.1f}")
    print(f"{'=' * 55}")
    print(f"\n  Dense sample:\n  {d_sample[:200]}")
    print(f"\n  MoE sample:\n  {m_sample[:200]}")


if __name__ == "__main__":
    main()
