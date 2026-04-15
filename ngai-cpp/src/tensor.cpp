#include "ngai.h"

namespace ngai {

void Tensor::fill(float val) {
    for (auto& v : data) v = val;
}

void rms_norm(float* x, const float* weight, size_t dim, float eps) {
    float sum_sq = 0.0f;
    for (size_t i = 0; i < dim; i++) {
        sum_sq += x[i] * x[i];
    }
    float rms = std::sqrt(sum_sq / static_cast<float>(dim) + eps);
    float inv_rms = 1.0f / rms;
    for (size_t i = 0; i < dim; i++) {
        x[i] = x[i] * inv_rms * weight[i];
    }
}

void silu_inplace(float* x, size_t n) {
    for (size_t i = 0; i < n; i++) {
        x[i] = x[i] / (1.0f + std::exp(-x[i]));
    }
}

void elementwise_mul(const float* a, const float* b, float* y, size_t n) {
    for (size_t i = 0; i < n; i++) {
        y[i] = a[i] * b[i];
    }
}

void elementwise_add(float* y, const float* x, size_t n) {
    for (size_t i = 0; i < n; i++) {
        y[i] += x[i];
    }
}

void softmax(float* x, size_t n) {
    float max_val = x[0];
    for (size_t i = 1; i < n; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        x[i] = std::exp(x[i] - max_val);
        sum += x[i];
    }
    float inv_sum = 1.0f / sum;
    for (size_t i = 0; i < n; i++) {
        x[i] *= inv_sum;
    }
}

void sigmoid_inplace(float* x, size_t n) {
    for (size_t i = 0; i < n; i++) {
        x[i] = 1.0f / (1.0f + std::exp(-x[i]));
    }
}

}  // namespace ngai
