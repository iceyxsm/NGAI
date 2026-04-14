"""CPU benchmark: NGAI block vs standard PyTorch.

Compares forward pass speed and memory of:
1. NGAI block (ternary weights + gated recurrence)
2. Standard nn.TransformerEncoderLayer (fp32 dense)

Run: uv run python experiments/benchmark_cpu.py
"""

import time

import torch
import torch.nn as nn

from ngai.core import NGAIBlock

WARMUP_RUNS = 5
BENCH_RUNS = 50
DIM = 256
SEQ_LEN = 128
BATCH = 1


def count_params(module: nn.Module) -> int:
    """Count total parameters in a module."""
    return sum(p.numel() for p in module.parameters())


def benchmark_ngai() -> dict[str, float]:
    """Benchmark NGAI block forward pass."""
    block = NGAIBlock(dim=DIM)
    block.eval()
    x = torch.randn(BATCH, SEQ_LEN, DIM)

    with torch.no_grad():
        for _ in range(WARMUP_RUNS):
            block(x)

        start = time.perf_counter()
        for _ in range(BENCH_RUNS):
            block(x)
        elapsed = time.perf_counter() - start

    params = count_params(block)
    tokens_per_sec = (BENCH_RUNS * SEQ_LEN) / elapsed

    mem_mb = sum(
        p.nelement() * p.element_size() for p in block.parameters()
    ) / (1024 * 1024)

    return {
        "name": "NGAI Block (ternary)",
        "params": params,
        "mem_mb": mem_mb,
        "total_sec": elapsed,
        "tokens_per_sec": tokens_per_sec,
    }


def benchmark_transformer() -> dict[str, float]:
    """Benchmark standard PyTorch TransformerEncoderLayer."""
    layer = nn.TransformerEncoderLayer(
        d_model=DIM, nhead=8, dim_feedforward=DIM * 4, batch_first=True
    )
    layer.eval()
    x = torch.randn(BATCH, SEQ_LEN, DIM)

    with torch.no_grad():
        for _ in range(WARMUP_RUNS):
            layer(x)

        start = time.perf_counter()
        for _ in range(BENCH_RUNS):
            layer(x)
        elapsed = time.perf_counter() - start

    params = count_params(layer)
    tokens_per_sec = (BENCH_RUNS * SEQ_LEN) / elapsed
    mem_mb = sum(
        p.nelement() * p.element_size() for p in layer.parameters()
    ) / (1024 * 1024)

    return {
        "name": "Transformer (fp32 dense)",
        "params": params,
        "mem_mb": mem_mb,
        "total_sec": elapsed,
        "tokens_per_sec": tokens_per_sec,
    }


def print_results(results: list[dict[str, float]]) -> None:
    """Print benchmark results as a table."""
    print(f"\n{'='*65}")
    print(f"  NGAI CPU Benchmark  (dim={DIM}, seq={SEQ_LEN}, batch={BATCH})")
    print(f"{'='*65}")
    print(f"  {'Model':<28} {'Params':>10} {'Mem(MB)':>8} {'Tok/s':>10}")
    print(f"  {'-'*28} {'-'*10} {'-'*8} {'-'*10}")
    for r in results:
        print(
            f"  {r['name']:<28} {r['params']:>10,} {r['mem_mb']:>8.2f}"
            f" {r['tokens_per_sec']:>10,.0f}"
        )
    print(f"{'='*65}")

    if len(results) >= 2:
        speedup = results[0]["tokens_per_sec"] / results[1]["tokens_per_sec"]
        mem_ratio = results[1]["mem_mb"] / results[0]["mem_mb"]
        print(f"\n  NGAI is {speedup:.1f}x {'faster' if speedup > 1 else 'slower'}")
        print(f"  NGAI uses {mem_ratio:.1f}x less memory (weight storage)")


if __name__ == "__main__":
    ngai_result = benchmark_ngai()
    transformer_result = benchmark_transformer()
    print_results([ngai_result, transformer_result])
