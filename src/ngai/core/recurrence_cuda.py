"""CUDA-accelerated gated recurrence kernel.

Replaces the Python for-loop with a fused CUDA kernel that runs
the entire recurrence on GPU in a single kernel launch.
Uses torch.utils.cpp_extension for JIT compilation.
"""

import torch
from torch import Tensor
from torch.autograd import Function
from torch.utils.cpp_extension import load_inline

CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void recurrence_fwd_kernel(
    const float* __restrict__ kv,
    const float* __restrict__ w,
    const float* __restrict__ state0,
    float* __restrict__ states,
    int batch_size, int seq_len, int dim
) {
    int b = blockIdx.x;
    int d = threadIdx.x + blockIdx.y * blockDim.x;
    if (b >= batch_size || d >= dim) return;
    float decay = w[d];
    float s = state0[b * dim + d];
    for (int t = 0; t < seq_len; t++) {
        int idx = (b * seq_len + t) * dim + d;
        s = decay * s + kv[idx];
        states[idx] = s;
    }
}

__global__ void recurrence_bwd_kernel(
    const float* __restrict__ grad_states,
    const float* __restrict__ w,
    float* __restrict__ grad_kv,
    float* __restrict__ grad_state0,
    int batch_size, int seq_len, int dim
) {
    int b = blockIdx.x;
    int d = threadIdx.x + blockIdx.y * blockDim.x;
    if (b >= batch_size || d >= dim) return;
    float decay = w[d];
    float gs = 0.0f;
    for (int t = seq_len - 1; t >= 0; t--) {
        int idx = (b * seq_len + t) * dim + d;
        gs += grad_states[idx];
        grad_kv[idx] = gs;
        gs *= decay;
    }
    grad_state0[b * dim + d] = gs;
}

torch::Tensor recurrence_fwd(
    torch::Tensor kv, torch::Tensor w, torch::Tensor state0
) {
    int batch = kv.size(0), seq_len = kv.size(1), dim = kv.size(2);
    auto states = torch::empty_like(kv);
    int threads = min(dim, 1024);
    int blocks_y = (dim + threads - 1) / threads;
    dim3 grid(batch, blocks_y);
    recurrence_fwd_kernel<<<grid, threads>>>(
        kv.data_ptr<float>(), w.data_ptr<float>(),
        state0.data_ptr<float>(), states.data_ptr<float>(),
        batch, seq_len, dim);
    return states;
}

std::vector<torch::Tensor> recurrence_bwd(
    torch::Tensor grad_states, torch::Tensor w
) {
    int batch = grad_states.size(0);
    int seq_len = grad_states.size(1);
    int dim = grad_states.size(2);
    auto grad_kv = torch::empty_like(grad_states);
    auto grad_s0 = torch::empty({batch, dim}, grad_states.options());
    int threads = min(dim, 1024);
    int blocks_y = (dim + threads - 1) / threads;
    dim3 grid(batch, blocks_y);
    recurrence_bwd_kernel<<<grid, threads>>>(
        grad_states.data_ptr<float>(), w.data_ptr<float>(),
        grad_kv.data_ptr<float>(), grad_s0.data_ptr<float>(),
        batch, seq_len, dim);
    return {grad_kv, grad_s0};
}
"""

CPP_SOURCE = r"""
torch::Tensor recurrence_fwd(
    torch::Tensor kv, torch::Tensor w, torch::Tensor state0);
std::vector<torch::Tensor> recurrence_bwd(
    torch::Tensor grad_states, torch::Tensor w);
"""

_cuda_module = None


def _get_cuda_module():  # noqa: ANN202
    """Lazy-load the CUDA extension."""
    global _cuda_module  # noqa: PLW0603
    if _cuda_module is None:
        _cuda_module = load_inline(
            name="ngai_recurrence",
            cpp_sources=[CPP_SOURCE],
            cuda_sources=[CUDA_SOURCE],
            functions=["recurrence_fwd", "recurrence_bwd"],
            verbose=False,
        )
    return _cuda_module


class CUDARecurrence(Function):
    """Custom autograd for CUDA recurrence."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        kv: Tensor, w: Tensor, state0: Tensor,
    ) -> Tensor:
        mod = _get_cuda_module()
        states = mod.recurrence_fwd(
            kv.contiguous(), w.contiguous(), state0.contiguous()
        )
        ctx.save_for_backward(w, states)
        return states

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_states: Tensor,
    ) -> tuple[Tensor, None, Tensor]:
        w, _states = ctx.saved_tensors
        mod = _get_cuda_module()
        grad_kv, grad_s0 = mod.recurrence_bwd(
            grad_states.contiguous(), w.contiguous()
        )
        return grad_kv, None, grad_s0


def cuda_recurrence(kv: Tensor, w: Tensor, state0: Tensor) -> Tensor:
    """Run gated recurrence on CUDA.

    Args:
        kv: Input values of shape (batch, seq_len, dim).
        w: Decay weights of shape (dim,).
        state0: Initial state of shape (batch, dim).

    Returns:
        All states of shape (batch, seq_len, dim).
    """
    return CUDARecurrence.apply(kv, w, state0)
