#pragma once
#include <vector>
#include <cstdint>

namespace mlwq {
namespace avx2 {
namespace ntt {
    std::vector<int16_t> poly_mul_ntt(const std::vector<int16_t>& a, const std::vector<int16_t>& b);
}
}
}