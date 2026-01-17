#ifndef MLWQ_AVX512_NOISE_H
#define MLWQ_AVX512_NOISE_H

#include <stdint.h>
#include "../common/structs.h"

void avx512_poly_getnoise_eta1(poly *r, const uint8_t seed[32], uint8_t nonce);
void avx512_polyvec_getnoise_eta1(poly_vec *r, const uint8_t seed[32], uint8_t nonce_base);

#endif
