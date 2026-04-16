"""Per-layer guided mutation using Forward-Forward goodness.

Instead of mutating all weights uniformly, this mutator:
1. Computes per-layer goodness scores (Forward-Forward style)
2. Focuses mutations on layers with LOW goodness (weak layers)
3. Preserves layers with HIGH goodness (strong layers)

This dramatically reduces the search space — instead of searching
over all N parameters simultaneously, we search over each layer's
K parameters independently, guided by local quality signals.
"""

import torch
import torch.nn as nn
from torch import Tensor

from ngai.evolve.evaluator import GoodnessEvaluator
from ngai.evolve.mutator import TernaryMutator


class LayerInfo:
    """Metadata for a single mutable layer group.

    Args:
        name: Parameter group name.
        start_idx: Start index in flat weight vector.
        end_idx: End index in flat weight vector.
        block_idx: Which model block this belongs to (-1 for non-block).
    """

    NON_BLOCK_IDX = -1

    def __init__(
        self, name: str, start_idx: int, end_idx: int, block_idx: int,
    ) -> None:
        self.name = name
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.block_idx = block_idx
        self.numel = end_idx - start_idx


class GuidedMutator:
    """Per-layer mutation guided by Forward-Forward goodness.

    Layers with low goodness get higher mutation rates.
    Layers with high goodness are mostly preserved.

    Args:
        model: The NGAI model.
        base_rate: Base mutation rate.
        goodness_scale: How much goodness affects mutation rate.
        device: Device for tensors.
    """

    BASE_RATE = 0.005
    GOODNESS_SCALE = 2.0
    MIN_RATE_FACTOR = 0.1
    MAX_RATE_FACTOR = 5.0

    def __init__(
        self,
        model: nn.Module,
        base_rate: float = BASE_RATE,
        goodness_scale: float = GOODNESS_SCALE,
        device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.base_rate = base_rate
        self.goodness_scale = goodness_scale
        self.device = device or torch.device("cpu")
        self.goodness_eval = GoodnessEvaluator()
        self.base_mutator = TernaryMutator(base_rate, self.device)
        self.layers = self._map_layers()
        self._layer_rates: list[float] = [base_rate] * len(self.layers)

    def _map_layers(self) -> list[LayerInfo]:
        """Map model parameters to layer groups with block indices."""
        layers: list[LayerInfo] = []
        offset = 0

        block_params: dict[int, list[tuple[str, int, int]]] = {}
        non_block_params: list[tuple[str, int, int]] = []

        for name, param in self.model.named_parameters():
            start = offset
            end = offset + param.numel()
            offset = end

            block_idx = self._extract_block_idx(name)
            if block_idx >= 0:
                if block_idx not in block_params:
                    block_params[block_idx] = []
                block_params[block_idx].append((name, start, end))
            else:
                non_block_params.append((name, start, end))

        for block_idx in sorted(block_params.keys()):
            params = block_params[block_idx]
            group_start = params[0][1]
            group_end = params[-1][2]
            layers.append(LayerInfo(
                name=f"block_{block_idx}",
                start_idx=group_start,
                end_idx=group_end,
                block_idx=block_idx,
            ))

        if non_block_params:
            layers.append(LayerInfo(
                name="non_block",
                start_idx=non_block_params[0][1],
                end_idx=non_block_params[-1][2],
                block_idx=LayerInfo.NON_BLOCK_IDX,
            ))

        return layers

    @staticmethod
    def _extract_block_idx(param_name: str) -> int:
        """Extract block index from parameter name like 'blocks.2.norm1.weight'.

        Returns:
            Block index, or -1 if not part of a block.
        """
        parts = param_name.split(".")
        if len(parts) >= 2 and parts[0] == "blocks" and parts[1].isdigit():
            return int(parts[1])
        return LayerInfo.NON_BLOCK_IDX

    def update_rates_from_goodness(
        self, layer_activations: list[Tensor],
    ) -> list[float]:
        """Update per-layer mutation rates based on goodness scores.

        Low goodness -> high mutation rate (layer needs more exploration).
        High goodness -> low mutation rate (layer is working well).

        Args:
            layer_activations: Per-block activation tensors.

        Returns:
            Updated mutation rates per layer.
        """
        if not layer_activations:
            return self._layer_rates

        goodness_scores = []
        for acts in layer_activations:
            flat_acts = acts.view(acts.shape[0], -1)
            score = self.goodness_eval.goodness(flat_acts).mean().item()
            goodness_scores.append(score)

        if not goodness_scores:
            return self._layer_rates

        mean_g = sum(goodness_scores) / len(goodness_scores)
        std_g = max(
            1e-8,
            (sum((g - mean_g) ** 2 for g in goodness_scores) / len(goodness_scores))
            ** 0.5,
        )

        block_layers = [ly for ly in self.layers if ly.block_idx >= 0]
        for layer, score in zip(block_layers, goodness_scores, strict=False):
            z_score = (score - mean_g) / std_g
            rate_factor = max(
                self.MIN_RATE_FACTOR,
                min(self.MAX_RATE_FACTOR, 1.0 - self.goodness_scale * z_score),
            )
            layer_idx = self.layers.index(layer)
            self._layer_rates[layer_idx] = self.base_rate * rate_factor

        return self._layer_rates

    def guided_mutate(
        self, flat_weights: Tensor, n_variants: int,
    ) -> Tensor:
        """Create variants with per-layer adaptive mutation rates.

        Each layer gets its own mutation rate based on goodness.
        Weak layers are mutated more aggressively, strong layers
        are mostly preserved.

        Args:
            flat_weights: Base weights of shape (total_params,).
            n_variants: Number of variants to create.

        Returns:
            Tensor of shape (n_variants, total_params).
        """
        variants = flat_weights.unsqueeze(0).expand(
            n_variants, -1,
        ).clone()

        for layer, rate in zip(self.layers, self._layer_rates, strict=True):
            chunk = variants[:, layer.start_idx : layer.end_idx]
            masks = torch.rand_like(chunk) < rate
            random_vals = torch.randint(
                -1, 2, chunk.shape, device=self.device,
            ).float()
            variants[:, layer.start_idx : layer.end_idx] = torch.where(
                masks, random_vals, chunk,
            )

        return variants
