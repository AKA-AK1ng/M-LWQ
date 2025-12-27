#pragma once
#include "../common/structs.hpp"

namespace mlwq {
namespace avx2 {
    int16_t positive_mod(int64_t val, int16_t q);
    poly poly_add(const poly& a, const poly& b);
    poly poly_sub(const poly& a, const poly& b);
    poly poly_mul_mod(const poly& a, const poly& b);
    poly_vec poly_matrix_vec_mul(const poly_matrix& A, const poly_vec& s);
    poly poly_vec_transpose_mul(const poly_vec& a_t, const poly_vec& b);
    poly_matrix poly_matrix_transpose(const poly_matrix& A);

    poly poly_quantize(const poly& val, const poly& d, int32_t P_param);
    poly_vec poly_vec_quantize(const poly_vec& val, const poly_vec& d, int32_t P_param);
    poly poly_dequantize(const poly& b, int32_t P_param);
    poly_vec poly_vec_dequantize(const poly_vec& b, int32_t P_param);

    poly poly_message_encode(const poly& m);
    poly poly_message_decode(const poly& val);
    
    bool check_poly_eq(const poly& a, const poly& b);
}
}