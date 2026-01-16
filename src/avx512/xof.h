#ifndef MLWQ_AVX512_XOF_H
#define MLWQ_AVX512_XOF_H

#include <stdint.h>
#include "../common/structs.h"

void avx512_xof_expand_matrix(poly_matrix *A, const uint8_t *seed);
void avx512_xof_expand_poly_vec(poly_vec *v, const uint8_t *seed, int32_t modulus);
void avx512_xof_expand_poly(poly *v, const uint8_t *seed, int32_t modulus);

#endif
