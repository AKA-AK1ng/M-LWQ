#ifndef MLWQ_AVX512_NTT_H
#define MLWQ_AVX512_NTT_H

#include <stdint.h>
#include "../common/structs.h"

void avx512_ntt(int16_t *r);
void avx512_invntt(int16_t *r);
void avx512_basemul(int16_t *r, const int16_t *a, const int16_t *b);
void avx512_reduce(int16_t *r);
void avx512_poly_mul_ntt(poly *res, const poly *a, const poly *b);

#endif
