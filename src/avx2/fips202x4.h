#ifndef FIPS202X4_H
#define FIPS202X4_H

#include <stddef.h>
#include <stdint.h>
#include <immintrin.h>

typedef struct {
  __m256i s[25];
} keccakx4_state;

// [FIX] 移除 KYBER_NAMESPACE 宏，防止链接名不匹配
// #define shake128x4_absorb_once KYBER_NAMESPACE(shake128x4_absorb_once)

void shake128x4_absorb_once(keccakx4_state *state,
                            const uint8_t *in0,
                            const uint8_t *in1,
                            const uint8_t *in2,
                            const uint8_t *in3,
                            size_t inlen);

void shake128x4_squeezeblocks(uint8_t *out0,
                              uint8_t *out1,
                              uint8_t *out2,
                              uint8_t *out3,
                              size_t nblocks,
                              keccakx4_state *state);

void shake256x4_absorb_once(keccakx4_state *state,
                            const uint8_t *in0,
                            const uint8_t *in1,
                            const uint8_t *in2,
                            const uint8_t *in3,
                            size_t inlen);

void shake256x4_squeezeblocks(uint8_t *out0,
                              uint8_t *out1,
                              uint8_t *out2,
                              uint8_t *out3,
                              size_t nblocks,
                              keccakx4_state *state);

#endif