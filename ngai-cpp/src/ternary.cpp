#include "ngai.h"

namespace ngai {

void TernaryWeight::set(size_t idx, int8_t val) {
    size_t byte_idx = idx / 4;
    size_t bit_offset = (idx % 4) * 2;
    uint8_t encoded;
    if (val == 0) encoded = 0b00;
    else if (val == 1) encoded = 0b01;
    else encoded = 0b10;  // val == -1

    data[byte_idx] &= ~(0b11 << bit_offset);  // clear
    data[byte_idx] |= (encoded << bit_offset);  // set
}

int8_t TernaryWeight::get(size_t idx) const {
    size_t byte_idx = idx / 4;
    size_t bit_offset = (idx % 4) * 2;
    uint8_t encoded = (data[byte_idx] >> bit_offset) & 0b11;
    if (encoded == 0b00) return 0;
    if (encoded == 0b01) return 1;
    return -1;
}

// The core operation: ternary matrix-vector multiply
// For each output element: y[i] = scale * sum_j(w[i,j] * x[j])
// Since w[i,j] is in {-1, 0, 1}, this is just conditional add/sub
void ternary_matvec(
    const TernaryWeight& w,
    const float* x,
    float* y,
    size_t out_dim,
    size_t in_dim
) {
    for (size_t i = 0; i < out_dim; i++) {
        float acc = 0.0f;
        size_t base = i * in_dim;

        // Process 4 weights at a time (one byte)
        size_t j = 0;
        for (; j + 3 < in_dim; j += 4) {
            size_t byte_idx = (base + j) / 4;
            uint8_t packed = w.data[byte_idx];

            // Unpack 4 ternary values and accumulate
            for (size_t k = 0; k < 4; k++) {
                uint8_t val = (packed >> (k * 2)) & 0b11;
                if (val == 0b01) acc += x[j + k];       // +1
                else if (val == 0b10) acc -= x[j + k];  // -1
                // val == 0b00: skip (multiply by 0)
            }
        }
        // Handle remaining weights
        for (; j < in_dim; j++) {
            int8_t val = w.get(base + j);
            if (val == 1) acc += x[j];
            else if (val == -1) acc -= x[j];
        }

        y[i] = acc * w.scale;
    }
}

}  // namespace ngai
