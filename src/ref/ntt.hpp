#pragma once
#include <vector>
#include <cstdint>
#include "../common/params.hpp"

namespace mlwq {
namespace ref {
namespace ntt {
    std::vector<int16_t> poly_mul_ntt(const std::vector<int16_t>& a, const std::vector<int16_t>& b);
}
}
}