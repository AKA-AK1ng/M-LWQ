#include "poly.hpp"
#include "ntt.hpp"
#include <immintrin.h>

namespace mlwq {
namespace avx2 {

// 复用 Ref 的标量实现 (如果有 AVX2 实现请替换)
int16_t positive_mod(int64_t val, int16_t q) {
    int16_t res = val % q;
    return (res < 0) ? (res + q) : static_cast<int16_t>(res);
}

// AVX2 Add
poly poly_add(const poly& a, const poly& b) {
    poly res(params::N);
    for (int i = 0; i < params::N; i += 16) { 
        __m256i va = _mm256_loadu_si256((__m256i*)&a[i]);
        __m256i vb = _mm256_loadu_si256((__m256i*)&b[i]);
        _mm256_storeu_si256((__m256i*)&res[i], _mm256_add_epi16(va, vb));
    }
    for(int i=0; i<params::N; ++i) if (res[i] >= params::Q) res[i] -= params::Q;
    return res;
}

// AVX2 Sub
poly poly_sub(const poly& a, const poly& b) {
    poly res(params::N);
    for (int i = 0; i < params::N; i += 16) { 
        __m256i va = _mm256_loadu_si256((__m256i*)&a[i]);
        __m256i vb = _mm256_loadu_si256((__m256i*)&b[i]);
        _mm256_storeu_si256((__m256i*)&res[i], _mm256_sub_epi16(va, vb));
    }
    for(int i=0; i<params::N; ++i) if (res[i] < 0) res[i] += params::Q;
    return res;
}

poly poly_mul_mod(const poly& a, const poly& b) {
    return ntt::poly_mul_ntt(a, b);
}

// 下面为了演示完整性，复用标量实现 (生产环境应用 AVX2 向量化)
poly poly_quantize(const poly& val, const poly& d, int32_t P_param) {
    poly res(params::N);
    const int32_t mask_P = P_param - 1; 
    for (size_t i = 0; i < params::N; ++i) {
        int32_t v = (int32_t)val[i] + d[i];
        int32_t floor_val = (v * P_param) / params::Q;
        res[i] = floor_val & mask_P;
    }
    return res;
}
poly poly_dequantize(const poly& b, int32_t P_param) {
    poly res(params::N);
    for (size_t i = 0; i < params::N; ++i) {
        int32_t v = (int32_t)b[i] * params::Q + (params::Q/2);
        res[i] = v / P_param;
    }
    return res;
}
poly poly_message_encode(const poly& m) {
    poly res(params::N);
    const int32_t scale = params::Q / 2;
    for (size_t i = 0; i < params::N; ++i) res[i] = (m[i] & 1) ? scale : 0;
    return res;
}
poly poly_message_decode(const poly& val) {
    poly res(params::N);
    int32_t limit_low = params::Q / 4;
    int32_t limit_high = 3 * params::Q / 4;
    for (size_t i = 0; i < params::N; ++i) res[i] = (val[i] > limit_low && val[i] < limit_high) ? 1 : 0;
    return res;
}
poly_vec poly_vec_quantize(const poly_vec& val, const poly_vec& d, int32_t P_param) {
    poly_vec res(val.size());
    for(size_t i=0; i<val.size(); ++i) res[i] = poly_quantize(val[i], d[i], P_param);
    return res;
}
poly_vec poly_vec_dequantize(const poly_vec& b, int32_t P_param) {
    poly_vec res(b.size());
    for(size_t i=0; i<b.size(); ++i) res[i] = poly_dequantize(b[i], P_param);
    return res;
}
poly_vec poly_matrix_vec_mul(const poly_matrix& A, const poly_vec& s) {
    poly_vec res(params::K);
    for(int i=0; i<params::K; ++i) {
        poly acc(params::N, 0);
        for(int j=0; j<params::K; ++j) acc = poly_add(acc, poly_mul_mod(A[i][j], s[j]));
        res[i] = acc;
    }
    return res;
}
poly poly_vec_transpose_mul(const poly_vec& a_t, const poly_vec& b) {
    poly res(params::N, 0);
    for(int i=0; i<params::K; ++i) res = poly_add(res, poly_mul_mod(a_t[i], b[i]));
    return res;
}
poly_matrix poly_matrix_transpose(const poly_matrix& A) {
    poly_matrix At(params::K, poly_vec(params::K));
    for(int i=0; i<params::K; ++i) for(int j=0; j<params::K; ++j) At[i][j] = A[j][i];
    return At;
}
bool check_poly_eq(const poly& a, const poly& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) if (a[i] != b[i]) return false;
    return true;
}

}
}