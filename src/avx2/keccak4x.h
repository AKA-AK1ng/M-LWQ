#ifndef AVX_KECCAK4X_H
#define AVX_KECCAK4X_H

#include <stdint.h>
#include <stddef.h>
#include <immintrin.h>

// 4路并行 Keccak 状态
typedef struct {
    __m256i s[25];
} keccak4x_state;

void shake128x4_absorb_once(keccak4x_state *state,
                            const uint8_t *in0, const uint8_t *in1,
                            const uint8_t *in2, const uint8_t *in3,
                            size_t inlen);

void shake128x4_squeezeblocks(uint8_t *out0, uint8_t *out1,
                              uint8_t *out2, uint8_t *out3,
                              size_t nblocks,
                              keccak4x_state *state);

#endif