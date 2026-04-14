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
