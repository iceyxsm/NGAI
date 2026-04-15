#include "ngai.h"

#ifdef __AVX2__
#include <immintrin.h>
#endif

namespace ngai {

void TernaryWeight::set(size_t idx, int8_t val) {
    size_t byte_idx = idx / 4;
    size_t bit_offset = (idx % 4) * 2;
    uint8_t encoded;
    if (val == 0) encoded = 0b00;
    else if (val == 1) encoded = 0b01;
    else encoded = 0b10;
    data[byte_idx] &= ~(0b11 << bit_offset);
    data[byte_idx] |= (encoded << bit_offset);
}

int8_t TernaryWeight::get(size_t idx) const {
    size_t byte_idx = idx / 4;
    size_t bit_offset = (idx % 4) * 2;
    uint8_t encoded = (data[byte_idx] >> bit_offset) & 0b11;
    if (encoded == 0b00) return 0;
    if (encoded == 0b01) return 1;
    return -1;
}

// Unpack one byte (4 ternary values) into 4 floats
// 00->0.0, 01->1.0, 10->-1.0
static inline void unpack4(uint8_t packed, float* out) {
    static constexpr float DECODE[4] = {0.0f, 1.0f, -1.0f, 0.0f};
    out[0] = DECODE[(packed >> 0) & 0b11];
    out[1] = DECODE[(packed >> 2) & 0b11];
    out[2] = DECODE[(packed >> 4) & 0b11];
    out[3] = DECODE[(packed >> 6) & 0b11];
}

// Scalar fallback
static void ternary_matvec_scalar(
    const TernaryWeight& w, const float* x, float* y,
    size_t out_dim, size_t in_dim
) {
    for (size_t i = 0; i < out_dim; i++) {
        float acc = 0.0f;
        size_t base = i * in_dim;
        size_t j = 0;
        for (; j + 3 < in_dim; j += 4) {
            alignas(16) float muls[4];
            unpack4(w.data[(base + j) / 4], muls);
            acc += muls[0] * x[j] + muls[1] * x[j+1] + muls[2] * x[j+2] + muls[3] * x[j+3];
        }
        for (; j < in_dim; j++) {
            int8_t val = w.get(base + j);
            if (val == 1) acc += x[j];
            else if (val == -1) acc -= x[j];
        }
        y[i] = acc * w.scale;
    }
}

#ifdef __AVX2__
// AVX2: unpack 8 ternary values from 2 bytes, FMA with input
static void ternary_matvec_avx2(
    const TernaryWeight& w, const float* x, float* y,
    size_t out_dim, size_t in_dim
) {
    static constexpr size_t SIMD_WIDTH = 8;

    for (size_t i = 0; i < out_dim; i++) {
        __m256 acc_vec = _mm256_setzero_ps();
        size_t base = i * in_dim;
        size_t j = 0;

        // Process 8 weights (2 bytes) at a time
        for (; j + SIMD_WIDTH - 1 < in_dim; j += SIMD_WIDTH) {
            // Load 8 input floats
            __m256 xv = _mm256_loadu_ps(x + j);

            // Unpack 2 bytes -> 8 float multipliers
            alignas(32) float muls[SIMD_WIDTH];
            size_t byte_base = (base + j) / 4;
            unpack4(w.data[byte_base], muls);
            unpack4(w.data[byte_base + 1], muls + 4);

            __m256 mv = _mm256_load_ps(muls);

            // acc += x * multiplier (FMA)
            acc_vec = _mm256_fmadd_ps(xv, mv, acc_vec);
        }

        // Horizontal sum
        __m128 hi = _mm256_extractf128_ps(acc_vec, 1);
        __m128 lo = _mm256_castps256_ps128(acc_vec);
        __m128 sum4 = _mm_add_ps(lo, hi);
        __m128 shuf = _mm_movehdup_ps(sum4);
        __m128 sum2 = _mm_add_ss(sum4, shuf);
        shuf = _mm_movehl_ps(shuf, sum2);
        __m128 sum1 = _mm_add_ss(sum2, shuf);
        float acc = _mm_cvtss_f32(sum1);

        // Tail
        for (; j < in_dim; j++) {
            int8_t val = w.get(base + j);
            if (val == 1) acc += x[j];
            else if (val == -1) acc -= x[j];
        }

        y[i] = acc * w.scale;
    }
}
#endif

void ternary_matvec(
    const TernaryWeight& w, const float* x, float* y,
    size_t out_dim, size_t in_dim
) {
#ifdef __AVX2__
    ternary_matvec_avx2(w, x, y, out_dim, in_dim);
#else
    ternary_matvec_scalar(w, x, y, out_dim, in_dim);
#endif
}

}  // namespace ngai
