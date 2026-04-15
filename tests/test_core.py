"""Tests for ngai.core — ternary quantization, linear layer, token mixer."""

import torch

from ngai.core import GatedRecurrence, TernaryLinear, ternary_quantize


class TestTernaryQuantize:
    """Tests for ternary weight quantization."""

    def test_output_values_are_ternary(self) -> None:
        w = torch.randn(64, 64)
        w_q = ternary_quantize(w)
        gamma = w.abs().mean()
        normalized = (w_q / gamma).round()
        unique = normalized.unique()
        for val in unique:
            assert val.item() in (-1.0, 0.0, 1.0)

    def test_shape_preserved(self) -> None:
        w = torch.randn(32, 128)
        w_q = ternary_quantize(w)
        assert w_q.shape == w.shape

    def test_gradient_flows_through(self) -> None:
        w = torch.randn(16, 16, requires_grad=True)
        w_q = ternary_quantize(w)
        loss = w_q.sum()
        loss.backward()
        assert w.grad is not None
        assert w.grad.shape == w.shape

    def test_zero_weights_produce_zero(self) -> None:
        w = torch.zeros(8, 8)
        w_q = ternary_quantize(w)
        assert torch.allclose(w_q, torch.zeros_like(w_q), atol=1e-6)


class TestTernaryLinear:
    """Tests for MatMul-free linear layer."""

    def test_forward_shape(self) -> None:
        layer = TernaryLinear(64, 32)
        x = torch.randn(4, 64)
        out = layer(x)
        assert out.shape == (4, 32)

    def test_batched_input(self) -> None:
        layer = TernaryLinear(128, 64)
        x = torch.randn(8, 10, 128)
        out = layer(x)
        assert out.shape == (8, 10, 64)

    def test_gradient_flows(self) -> None:
        layer = TernaryLinear(32, 16)
        x = torch.randn(2, 32)
        out = layer(x)
        out.sum().backward()
        assert layer.weight.grad is not None

    def test_with_bias(self) -> None:
        layer = TernaryLinear(32, 16, bias=True)
        x = torch.randn(2, 32)
        out = layer(x)
        assert out.shape == (2, 16)
        assert layer.bias is not None


class TestGatedRecurrence:
    """Tests for RWKV-style token mixer."""

    def test_output_shape(self) -> None:
        mixer = GatedRecurrence(dim=64)
        x = torch.randn(2, 10, 64)
        out, state = mixer(x)
        assert out.shape == (2, 10, 64)
        assert state.shape == (2, 64)

    def test_state_continuity(self) -> None:
        mixer = GatedRecurrence(dim=32)
        x1 = torch.randn(1, 5, 32)
        x2 = torch.randn(1, 5, 32)
        _, state1 = mixer(x1)
        out_cont, _ = mixer(x2, state=state1)
        out_fresh, _ = mixer(x2)
        assert not torch.allclose(out_cont, out_fresh, atol=1e-5)

    def test_gradient_flows(self) -> None:
        mixer = GatedRecurrence(dim=16)
        x = torch.randn(1, 3, 16)
        out, _ = mixer(x)
        out.sum().backward()
        assert mixer.receptance.weight.grad is not None
        assert mixer.key.weight.grad is not None
        assert mixer.value.weight.grad is not None

    def test_single_token(self) -> None:
        mixer = GatedRecurrence(dim=32)
        x = torch.randn(1, 1, 32)
        out, state = mixer(x)
        assert out.shape == (1, 1, 32)
        assert state.shape == (1, 32)


from ngai.core import ExpertRouter, MoEChannelMixer  # noqa: E402


class TestExpertRouter:
    """Tests for MoE routing."""

    def test_output_shapes(self) -> None:
        router = ExpertRouter(dim=32, n_experts=8, top_k=2)
        x = torch.randn(10, 32)
        weights, indices, loss = router(x)
        assert weights.shape == (10, 2)
        assert indices.shape == (10, 2)
        assert loss.shape == ()

    def test_weights_sum_to_one(self) -> None:
        router = ExpertRouter(dim=32, n_experts=8, top_k=2)
        x = torch.randn(10, 32)
        weights, _, _ = router(x)
        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_indices_in_range(self) -> None:
        router = ExpertRouter(dim=32, n_experts=8, top_k=2)
        x = torch.randn(10, 32)
        _, indices, _ = router(x)
        assert (indices >= 0).all()
        assert (indices < 8).all()


class TestMoEChannelMixer:
    """Tests for MoE channel mixer."""

    def test_output_shape(self) -> None:
        moe = MoEChannelMixer(dim=32, n_shared=1, n_routed=4, top_k=2)
        x = torch.randn(2, 5, 32)
        out, loss = moe(x)
        assert out.shape == (2, 5, 32)
        assert loss.shape == ()

    def test_gradient_flows(self) -> None:
        moe = MoEChannelMixer(dim=16, n_shared=1, n_routed=4, top_k=1)
        x = torch.randn(1, 3, 16)
        out, loss = moe(x)
        (out.sum() + loss).backward()
        assert moe.router.gate.weight.grad is not None
        assert moe.shared_experts[0].gate.weight.grad is not None


from ngai.core.novel_routing import (  # noqa: E402
    AdaptiveRouter,
    CrossLayerExpertPool,
    StateAwareRouter,
)


class TestStateAwareRouter:
    """Tests for state-aware routing."""

    def test_output_shapes(self) -> None:
        router = StateAwareRouter(dim=32, n_experts=8, top_k=2)
        x = torch.randn(10, 32)
        state = torch.randn(10, 32)
        weights, indices, _loss = router(x, state)
        assert weights.shape == (10, 2)
        assert indices.shape == (10, 2)

    def test_works_without_state(self) -> None:
        router = StateAwareRouter(dim=32, n_experts=8, top_k=2)
        x = torch.randn(10, 32)
        weights, _indices, _loss = router(x)
        assert weights.shape == (10, 2)

    def test_state_changes_routing(self) -> None:
        router = StateAwareRouter(dim=32, n_experts=4, top_k=2)
        x = torch.randn(5, 32)
        s1 = torch.randn(5, 32)
        s2 = torch.randn(5, 32) * 10
        _, idx1, _ = router(x, s1)
        _, idx2, _ = router(x, s2)
        assert not torch.equal(idx1, idx2)


class TestAdaptiveRouter:
    """Tests for adaptive sparsity routing."""

    def test_output_shapes(self) -> None:
        router = AdaptiveRouter(dim=32, n_experts=8, max_k=4)
        x = torch.randn(10, 32)
        weights, indices, _loss, avg_k = router(x)
        assert weights.shape == (10, 4)
        assert indices.shape == (10, 4)
        assert avg_k.shape == ()

    def test_avg_k_in_range(self) -> None:
        router = AdaptiveRouter(dim=32, n_experts=8, max_k=4)
        x = torch.randn(20, 32)
        _, _, _, avg_k = router(x)
        assert 1.0 <= avg_k.item() <= 4.0

    def test_gradient_flows(self) -> None:
        router = AdaptiveRouter(dim=16, n_experts=4, max_k=2)
        x = torch.randn(5, 16)
        weights, _, loss, _ = router(x)
        (weights.sum() + loss).backward()
        assert router.gate.weight.grad is not None


class TestCrossLayerExpertPool:
    """Tests for shared expert pool."""

    def test_output_shape(self) -> None:
        pool = CrossLayerExpertPool(dim=32, n_experts=4)
        x = torch.randn(10, 32)
        weights = torch.ones(10, 2) * 0.5
        indices = torch.randint(0, 4, (10, 2))
        out = pool(x, weights, indices)
        assert out.shape == (10, 32)

    def test_gradient_flows(self) -> None:
        pool = CrossLayerExpertPool(dim=16, n_experts=4)
        x = torch.randn(5, 16)
        weights = torch.ones(5, 1) * 1.0
        indices = torch.zeros(5, 1, dtype=torch.long)
        out = pool(x, weights, indices)
        out.sum().backward()
        assert pool.experts[0].gate.weight.grad is not None
