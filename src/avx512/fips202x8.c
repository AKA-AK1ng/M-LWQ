#include <stddef.h>
#include <stdint.h>
#include <immintrin.h>
#include "fips202x8.h"

#define SHAKE128_RATE 168
#define SHAKE256_RATE 136

static void keccakx8_absorb_once(
    KeccakP1600times8_SIMD512_states *state,
    unsigned int r,
    const uint8_t *in0,
    const uint8_t *in1,
    const uint8_t *in2,
    const uint8_t *in3,
    const uint8_t *in4,
    const uint8_t *in5,
    const uint8_t *in6,
    const uint8_t *in7,
    size_t inlen,
    uint8_t p
) {
    size_t i;
    uint64_t pos = 0;
    __m512i t;
    __m512i idx;

    for (i = 0; i < 25; i++) {
        state->A[i] = _mm512_setzero_si512();
    }

    idx = _mm512_set_epi64((long long)in7,
                           (long long)in6,
                           (long long)in5,
                           (long long)in4,
                           (long long)in3,
                           (long long)in2,
                           (long long)in1,
                           (long long)in0);

    while (inlen >= r) {
        for (i = 0; i < r / 8; i++) {
            t = _mm512_i64gather_epi64(idx, (const void *)pos, 1);
            state->A[i] = _mm512_xor_si512(state->A[i], t);
        }

        KeccakP1600times8_AVX512_PermuteAll_24rounds(state);
        inlen -= r;
        pos += r;
    }

    for (i = 0; i < inlen / 8; i++) {
        t = _mm512_i64gather_epi64(idx, (const void *)pos, 1);
        state->A[i] = _mm512_xor_si512(state->A[i], t);
    }

    const uint8_t *ins[8] = {in0 + pos, in1 + pos, in2 + pos, in3 + pos,
                             in4 + pos, in5 + pos, in6 + pos, in7 + pos};
    size_t rem = inlen % 8;

    for (int stream = 0; stream < 8; stream++) {
        for (i = 0; i < rem; i++) {
            uint64_t lane_idx = (i + inlen - rem) / 8;
            int byte_in_lane = (int)((i + inlen - rem) % 8);
            uint64_t val = (uint64_t)ins[stream][i];
            ((uint64_t *)&state->A[lane_idx])[stream] ^= (val << (8 * byte_in_lane));
        }

        uint64_t lane_idx = inlen / 8;
        int byte_in_lane = (int)(inlen % 8);
        ((uint64_t *)&state->A[lane_idx])[stream] ^= ((uint64_t)p << (8 * byte_in_lane));

        lane_idx = (r - 1) / 8;
        byte_in_lane = (int)((r - 1) % 8);
        ((uint64_t *)&state->A[lane_idx])[stream] ^= (0x80ULL << (8 * byte_in_lane));
    }

    KeccakP1600times8_AVX512_PermuteAll_24rounds(state);
}

static void keccakx8_squeezeblocks(
    uint8_t *out0,
    uint8_t *out1,
    uint8_t *out2,
    uint8_t *out3,
    uint8_t *out4,
    uint8_t *out5,
    uint8_t *out6,
    uint8_t *out7,
    size_t nblocks,
    unsigned int r,
    KeccakP1600times8_SIMD512_states *state
) {
    size_t i;

    while (nblocks > 0) {
        for (i = 0; i < r / 8; i++) {
            __m512i t = state->A[i];
            uint64_t v[8];
            _mm512_storeu_si512((__m512i *)v, t);
            ((uint64_t *)out0)[i] = v[0];
            ((uint64_t *)out1)[i] = v[1];
            ((uint64_t *)out2)[i] = v[2];
            ((uint64_t *)out3)[i] = v[3];
            ((uint64_t *)out4)[i] = v[4];
            ((uint64_t *)out5)[i] = v[5];
            ((uint64_t *)out6)[i] = v[6];
            ((uint64_t *)out7)[i] = v[7];
        }

        KeccakP1600times8_AVX512_PermuteAll_24rounds(state);

        out0 += r;
        out1 += r;
        out2 += r;
        out3 += r;
        out4 += r;
        out5 += r;
        out6 += r;
        out7 += r;
        nblocks--;
    }
}

void shake128x8_absorb_once(keccakx8_state *state,
                            const uint8_t *in0,
                            const uint8_t *in1,
                            const uint8_t *in2,
                            const uint8_t *in3,
                            const uint8_t *in4,
                            const uint8_t *in5,
                            const uint8_t *in6,
                            const uint8_t *in7,
                            size_t inlen) {
    keccakx8_absorb_once(&state->s, SHAKE128_RATE, in0, in1, in2, in3, in4, in5, in6, in7, inlen, 0x1F);
}

void shake128x8_squeezeblocks(uint8_t *out0,
                              uint8_t *out1,
                              uint8_t *out2,
                              uint8_t *out3,
                              uint8_t *out4,
                              uint8_t *out5,
                              uint8_t *out6,
                              uint8_t *out7,
                              size_t nblocks,
                              keccakx8_state *state) {
    keccakx8_squeezeblocks(out0, out1, out2, out3, out4, out5, out6, out7, nblocks, SHAKE128_RATE, &state->s);
}

void shake256x8_absorb_once(keccakx8_state *state,
                            const uint8_t *in0,
                            const uint8_t *in1,
                            const uint8_t *in2,
                            const uint8_t *in3,
                            const uint8_t *in4,
                            const uint8_t *in5,
                            const uint8_t *in6,
                            const uint8_t *in7,
                            size_t inlen) {
    keccakx8_absorb_once(&state->s, SHAKE256_RATE, in0, in1, in2, in3, in4, in5, in6, in7, inlen, 0x1F);
}

void shake256x8_squeezeblocks(uint8_t *out0,
                              uint8_t *out1,
                              uint8_t *out2,
                              uint8_t *out3,
                              uint8_t *out4,
                              uint8_t *out5,
                              uint8_t *out6,
                              uint8_t *out7,
                              size_t nblocks,
                              keccakx8_state *state) {
    keccakx8_squeezeblocks(out0, out1, out2, out3, out4, out5, out6, out7, nblocks, SHAKE256_RATE, &state->s);
}
