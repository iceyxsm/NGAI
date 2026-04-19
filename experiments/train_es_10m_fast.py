"""Fast layer-wise ES at 10M scale — fp16 eval + per-block groups.

Two speedups over train_es_10m.py:
1. fp16 forward passes (2x faster on T4 tensor cores)
2. Per-block groups (8 groups instead of 58, fewer cycles)

Same algorithm, same math. Just faster execution.

Run on GPU: python experiments/train_es_10m_fast.py
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


class Fp16LayerwiseES(LayerwiseESTrainer):
    """Layer-wise ES with fp16 forward passes for 2x speed on T4."""

    @torch.no_grad()
    def _forward_loss(self, x, y):
        """Forward pass in fp16 for tensor core acceleration."""
        self.model.eval()
        with torch.amp.autocast("cuda"):
            logits = self.model(x)
            if isinstance(logits, tuple):
                logits = logits[0]
            vocab = logits.shape[-1]
            return torch.nn.functional.cross_entropy(
                logits.view(-1, vocab), y.view(-1),
            )

    @torch.no_grad()
    def _suffix_forward_loss(self, cached_x, from_block, y):
        """Suffix forward pass in fp16."""
        with torch.amp.autocast("cuda"):
            x = cached_x
            for i in range(from_block, len(self.model.blocks)):
                x, _ = self.model.blocks[i](x, None)
            x = self.model.norm(x)
            logits = self.model.head(x)
            if isinstance(logits, tuple):
                logits = logits[0]
            vocab = logits.shape[-1]
            return torch.nn.functional.cross_entropy(
                logits.view(-1, vocab), y.view(-1),
            )

    @torch.no_grad()
    def _compute_prefix(self, input_ids, up_to_block):
        """Prefix computation in fp16."""
        with torch.amp.autocast("cuda"):
            x = self.model.embedding(input_ids)
            for i in range(min(up_to_block, len(self.model.blocks))):
                x, _ = self.model.blocks[i](x, None)
            return x


class BlockGroupES(Fp16LayerwiseES):
    """Layer-wise ES with per-block grouping (8 groups instead of 58).

    Consolidates all sublayers within a block into one group.
    Each step perturbs an entire block (~1M params) at once.
    """

    def _build_param_groups(self):
        """Group all params within each block together."""
        groups = []

        emb_params = list(self.model.embedding.parameters())
        emb_numel = sum(p.numel() for p in emb_params)
        groups.append({
            "name": "embedding",
            "params": emb_params,
            "numel": emb_numel,
        })

        for bi, block in enumerate(self.model.blocks):
            block_params = list(block.parameters())
            block_numel = sum(p.numel() for p in block_params)
            groups.append({
                "name": f"blocks.{bi}",
                "params": block_params,
                "numel": block_numel,
            })

        head_params = list(self.model.head.parameters())
        norm_params = list(self.model.norm.parameters())
        tail_params = norm_params + head_params
        tail_numel = sum(p.numel() for p in tail_params)
        groups.append({
            "name": "head",
            "params": tail_params,
            "numel": tail_numel,
        })

        return groups


def main() -> None:
    """Run fast layer-wise ES at 10M scale."""
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
    print("  Block-group ES + fp16 (no backprop)")
    print(f"  Model: {params:,} params, dim={DIM}, layers={N_LAYERS}")
    print(f"  Pop={POP_SIZE}, sigma={SIGMA}, lr={LR}")
    print("=" * 55)

    trainer = BlockGroupES(
        model, pop_size=POP_SIZE, sigma=SIGMA,
        lr=LR, momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY, device=DEVICE,
    )
    losses = trainer.train(loader, STEPS)

    window = min(100, len(losses))
    final = sum(losses[-window:]) / window

    print("\n" + "=" * 55)
    print("  10M FAST LAYER-WISE ES RESULTS")
    print("=" * 55)
    print(f"  Final avg loss:      {final:.4f}")
    print(f"  Best loss:           {trainer.best_loss:.4f}")
    print(f"  Backprop reference:  ~{BACKPROP_REFERENCE_LOSS}")
    print(f"  Gap:                 {final / BACKPROP_REFERENCE_LOSS:.1f}x")
    print("=" * 55)


if __name__ == "__main__":
    main()
