#pragma once
#include <stdint.h>
#include <immintrin.h>

struct Keccak4x_State {
    __m256i s[25];
};

void KeccakP1600_times4_AVX2(__m256i *s);

void shake128x4_absorb_once(Keccak4x_State *state,
                            const uint8_t *in0, const uint8_t *in1,
                            const uint8_t *in2, const uint8_t *in3,
                            size_t inlen);

void shake128x4_squeezeblocks(uint8_t *out0, uint8_t *out1,
                              uint8_t *out2, uint8_t *out3,
                              size_t nblocks, Keccak4x_State *state);