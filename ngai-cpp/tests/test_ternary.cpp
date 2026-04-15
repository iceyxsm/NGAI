#include "ngai.h"
#include <cassert>
#include <cmath>

static constexpr float TOLERANCE = 1e-5f;

void test_ternary_packing() {
    ngai::TernaryWeight w(8, 1.0f);

    w.set(0, 1);
    w.set(1, -1);
    w.set(2, 0);
    w.set(3, 1);
    w.set(4, -1);
    w.set(5, 0);
    w.set(6, -1);
    w.set(7, 1);

    assert(w.get(0) == 1);
    assert(w.get(1) == -1);
    assert(w.get(2) == 0);
    assert(w.get(3) == 1);
    assert(w.get(4) == -1);
    assert(w.get(5) == 0);
    assert(w.get(6) == -1);
    assert(w.get(7) == 1);

    std::cout << "PASS: ternary_packing\n";
}

void test_ternary_matvec() {
    // 2x3 matrix: [[1, -1, 0], [0, 1, -1]]
    ngai::TernaryWeight w(6, 1.0f);
    w.set(0, 1);  w.set(1, -1); w.set(2, 0);
    w.set(3, 0);  w.set(4, 1);  w.set(5, -1);

    float x[3] = {2.0f, 3.0f, 4.0f};
    float y[2] = {0.0f, 0.0f};

    ngai::ternary_matvec(w, x, y, 2, 3);

    // y[0] = 1*2 + (-1)*3 + 0*4 = -1
    // y[1] = 0*2 + 1*3 + (-1)*4 = -1
    assert(std::abs(y[0] - (-1.0f)) < TOLERANCE);
    assert(std::abs(y[1] - (-1.0f)) < TOLERANCE);

    std::cout << "PASS: ternary_matvec\n";
}

void test_ternary_matvec_with_scale() {
    ngai::TernaryWeight w(4, 0.5f);
    w.set(0, 1); w.set(1, 1);
    w.set(2, -1); w.set(3, -1);

    float x[2] = {1.0f, 1.0f};
    float y[2] = {0.0f, 0.0f};

    ngai::ternary_matvec(w, x, y, 2, 2);

    // y[0] = 0.5 * (1+1) = 1.0
    // y[1] = 0.5 * (-1-1) = -1.0
    assert(std::abs(y[0] - 1.0f) < TOLERANCE);
    assert(std::abs(y[1] - (-1.0f)) < TOLERANCE);

    std::cout << "PASS: ternary_matvec_with_scale\n";
}

void test_rms_norm() {
    float x[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float w[4] = {1.0f, 1.0f, 1.0f, 1.0f};

    ngai::rms_norm(x, w, 4);

    // RMS = sqrt((1+4+9+16)/4) = sqrt(7.5) ~ 2.7386
    float rms = std::sqrt(7.5f);
    assert(std::abs(x[0] - 1.0f/rms) < TOLERANCE);

    std::cout << "PASS: rms_norm\n";
}

int main() {
    std::cout << "NGAI C++ Tests\n";
    std::cout << "==============\n";

    test_ternary_packing();
    test_ternary_matvec();
    test_ternary_matvec_with_scale();
    test_rms_norm();

    std::cout << "\nAll tests passed!\n";
    return 0;
}
