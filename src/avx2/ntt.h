#ifndef AVX_NTT_H
#define AVX_NTT_H

#include <stdint.h>
#include <immintrin.h>
#include "../common/structs.h"

void avx_ntt(int16_t *r);
void avx_invntt(int16_t *r);
void avx_poly_mul_ntt(poly *res, const poly *a, const poly *b);

// AVX2 基础运算供其他模块使用
__m256i avx_montgomery_reduce(__m256i a);
__m256i avx_fqmul(__m256i a, __m256i b);

#endif