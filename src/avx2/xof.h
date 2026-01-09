#ifndef AVX_XOF_H
#define AVX_XOF_H

#include "../common/structs.h"

void avx_xof_expand_matrix(poly_matrix *A, const uint8_t *seed);

// [FIX] 增加 modulus 参数
void avx_xof_expand_poly_vec(poly_vec *v, const uint8_t *seed, int32_t modulus);
void avx_xof_expand_poly(poly *v, const uint8_t *seed, int32_t modulus);

#endif
