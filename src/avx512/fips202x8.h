#ifndef FIPS202X8_H
#define FIPS202X8_H

#include <stddef.h>
#include <stdint.h>
#include <immintrin.h>
#include "keccak8x/KeccakP-1600-times8-AVX512.h"

typedef struct {
  KeccakP1600times8_SIMD512_states s;
} keccakx8_state;

void shake128x8_absorb_once(keccakx8_state *state,
                            const uint8_t *in0,
                            const uint8_t *in1,
                            const uint8_t *in2,
                            const uint8_t *in3,
                            const uint8_t *in4,
                            const uint8_t *in5,
                            const uint8_t *in6,
                            const uint8_t *in7,
                            size_t inlen);

void shake128x8_squeezeblocks(uint8_t *out0,
                              uint8_t *out1,
                              uint8_t *out2,
                              uint8_t *out3,
                              uint8_t *out4,
                              uint8_t *out5,
                              uint8_t *out6,
                              uint8_t *out7,
                              size_t nblocks,
                              keccakx8_state *state);

void shake256x8_absorb_once(keccakx8_state *state,
                            const uint8_t *in0,
                            const uint8_t *in1,
                            const uint8_t *in2,
                            const uint8_t *in3,
                            const uint8_t *in4,
                            const uint8_t *in5,
                            const uint8_t *in6,
                            const uint8_t *in7,
                            size_t inlen);

void shake256x8_squeezeblocks(uint8_t *out0,
                              uint8_t *out1,
                              uint8_t *out2,
                              uint8_t *out3,
                              uint8_t *out4,
                              uint8_t *out5,
                              uint8_t *out6,
                              uint8_t *out7,
                              size_t nblocks,
                              keccakx8_state *state);

#endif
