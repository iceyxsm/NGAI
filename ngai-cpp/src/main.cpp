#include "ngai.h"
#include <random>

static constexpr size_t DIM = 4096;
static constexpr size_t WARMUP = 10;
static constexpr size_t BENCH_ITERS = 1000;

int main() {
    std::cout << "NGAI C++ Inference Engine - Ternary Benchmark\n";
    std::cout << "=============================================\n\n";

    // Create a random ternary weight matrix (DIM x DIM)
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(-1, 1);

    ngai::TernaryWeight weights(DIM * DIM, 0.5f);
    for (size_t i = 0; i < DIM * DIM; i++) {
        weights.set(i, static_cast<int8_t>(dist(rng)));
    }

    // Create random input vector
    std::vector<float> x(DIM);
    std::vector<float> y(DIM);
    std::normal_distribution<float> ndist(0.0f, 1.0f);
    for (auto& v : x) v = ndist(rng);

    // Memory stats
    size_t weight_bytes = weights.size_bytes();
    size_t fp32_bytes = DIM * DIM * sizeof(float);
    std::cout << "Matrix size: " << DIM << " x " << DIM << "\n";
    std::cout << "Ternary weight memory: " << weight_bytes / 1024 << " KB\n";
    std::cout << "FP32 equivalent:       " << fp32_bytes / 1024 << " KB\n";
    std::cout << "Compression ratio:     " << static_cast<float>(fp32_bytes) / weight_bytes << "x\n\n";

    // Warmup
    for (size_t i = 0; i < WARMUP; i++) {
        ngai::ternary_matvec(weights, x.data(), y.data(), DIM, DIM);
    }

    // Benchmark
    ngai::Timer timer;
    for (size_t i = 0; i < BENCH_ITERS; i++) {
        ngai::ternary_matvec(weights, x.data(), y.data(), DIM, DIM);
    }
    double elapsed = timer.elapsed_ms();

    double ms_per_op = elapsed / BENCH_ITERS;
    double ops_per_sec = 1000.0 / ms_per_op;
    double gops = (2.0 * DIM * DIM * ops_per_sec) / 1e9;

    std::cout << "Benchmark: " << BENCH_ITERS << " matvec operations\n";
    std::cout << "Total time:    " << elapsed << " ms\n";
    std::cout << "Per operation: " << ms_per_op << " ms\n";
    std::cout << "Throughput:    " << ops_per_sec << " ops/sec\n";
    std::cout << "GOPS:          " << gops << " (ternary accumulations/sec)\n\n";

    // Estimate tokens/sec for a 4-layer model
    static constexpr size_t N_LAYERS = 4;
    static constexpr size_t MATVECS_PER_LAYER = 8;  // r,k,v,o + gate,up,down + router
    double ms_per_token = ms_per_op * N_LAYERS * MATVECS_PER_LAYER;
    double tokens_per_sec = 1000.0 / ms_per_token;
    std::cout << "Estimated inference (4-layer, dim=" << DIM << "):\n";
    std::cout << "  " << ms_per_token << " ms/token\n";
    std::cout << "  " << tokens_per_sec << " tokens/sec\n";

    return 0;
}
