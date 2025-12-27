#pragma once
#include "structs.hpp"

namespace mlwq {
    poly random_poly_uniform(int32_t modulus);
    poly random_poly_eta(int32_t eta);
    poly_vec random_poly_vec_eta(int32_t k, int32_t eta);
}