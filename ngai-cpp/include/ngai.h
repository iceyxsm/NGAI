#pragma once
#include <cstdint>
#include <cstddef>
#include <vector>
#include <string>
#include <cmath>
#include <cassert>
#include <fstream>
#include <iostream>
#include <chrono>

namespace ngai {

// Packed ternary weights: 2 bits per weight
// 00 = 0, 01 = +1, 10 = -1, 11 = unused
struct TernaryWeight {
    std::vector<uint8_t> data;  // packed 4 weights per byte
    size_t n_weights;
    float scale;  // absmean scaling factor

    TernaryWeight() : n_weights(0), scale(1.0f) {}
    TernaryWeight(size_t n, float s) : data((n + 3) / 4), n_weights(n), scale(s) {}

    void set(size_t idx, int8_t val);  // val in {-1, 0, 1}
    int8_t get(size_t idx) const;
    size_t size_bytes() const { return data.size(); }
};

// Simple float tensor (for activations)
struct Tensor {
    std::vector<float> data;
    size_t rows, cols;

    Tensor() : rows(0), cols(0) {}
    Tensor(size_t r, size_t c) : data(r * c, 0.0f), rows(r), cols(c) {}

    float& at(size_t r, size_t c) { return data[r * cols + c]; }
    float at(size_t r, size_t c) const { return data[r * cols + c]; }
    float* row_ptr(size_t r) { return data.data() + r * cols; }
    const float* row_ptr(size_t r) const { return data.data() + r * cols; }
    size_t numel() const { return data.size(); }
    void fill(float val);
};

// Ternary matrix-vector multiply: y = W * x
// W is ternary (packed), x and y are float vectors
void ternary_matvec(
    const TernaryWeight& w,
    const float* x,
    float* y,
    size_t out_dim,
    size_t in_dim
);

// RMS normalization in-place
void rms_norm(float* x, const float* weight, size_t dim, float eps = 1e-6f);

// SiLU activation in-place
void silu_inplace(float* x, size_t n);

// Element-wise multiply: y = a * b
void elementwise_mul(const float* a, const float* b, float* y, size_t n);

// Element-wise add: y += x
void elementwise_add(float* y, const float* x, size_t n);

// Softmax in-place
void softmax(float* x, size_t n);

// Sigmoid in-place
void sigmoid_inplace(float* x, size_t n);

// Timing utility
struct Timer {
    std::chrono::high_resolution_clock::time_point start;
    Timer() : start(std::chrono::high_resolution_clock::now()) {}
    double elapsed_ms() const {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(now - start).count();
    }
};

}  // namespace ngai
