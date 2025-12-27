#include "keccak4x.hpp"
#include <immintrin.h>
#include <string.h>

#define ROL64(x, n) _mm256_xor_si256(_mm256_slli_epi64(x, n), _mm256_srli_epi64(x, 64 - n))

static const uint64_t KeccakF_RoundConstants[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL,
    0x8000000080008000ULL, 0x000000000000808bULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008aULL,
    0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
    0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL,
    0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800aULL, 0x800000008000000aULL, 0x8000000080008081ULL,
    0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL
};

void KeccakP1600_times4_AVX2(__m256i *s) {
    for (int r = 0; r < 24; ++r) {
        __m256i BC[5];
        for (int i = 0; i < 5; ++i) BC[i] = _mm256_xor_si256(_mm256_xor_si256(s[i], s[i+5]), _mm256_xor_si256(_mm256_xor_si256(s[i+10], s[i+15]), s[i+20]));
        for (int i = 0; i < 5; ++i) {
            __m256i t = _mm256_xor_si256(BC[(i+4)%5], ROL64(BC[(i+1)%5], 1));
            for (int j = 0; j < 25; j += 5) s[j+i] = _mm256_xor_si256(s[j+i], t);
        }
        __m256i t, tmp;
        t = s[1];
        #define PI_RHO(o, i, r) tmp = s[i]; s[i] = ROL64(t, r); t = tmp;
        PI_RHO(1, 10, 1); PI_RHO(10, 7, 3); PI_RHO(7, 11, 6); PI_RHO(11, 17, 10);
        PI_RHO(17, 18, 15); PI_RHO(18, 3, 21); PI_RHO(3, 5, 28); PI_RHO(5, 16, 36);
        PI_RHO(16, 8, 45); PI_RHO(8, 21, 55); PI_RHO(21, 24, 2); PI_RHO(24, 4, 14);
        PI_RHO(4, 15, 27); PI_RHO(15, 23, 41); PI_RHO(23, 19, 56); PI_RHO(19, 13, 8);
        PI_RHO(13, 12, 25); PI_RHO(12, 2, 43); PI_RHO(2, 20, 62); PI_RHO(20, 14, 18);
        PI_RHO(14, 22, 39); PI_RHO(22, 9, 61); PI_RHO(9, 6, 20); PI_RHO(6, 1, 44);
        for (int j = 0; j < 25; j += 5) {
            __m256i s0 = s[j+0], s1 = s[j+1], s2 = s[j+2], s3 = s[j+3], s4 = s[j+4];
            s[j+0] = _mm256_xor_si256(s0, _mm256_andnot_si256(s1, s2));
            s[j+1] = _mm256_xor_si256(s1, _mm256_andnot_si256(s2, s3));
            s[j+2] = _mm256_xor_si256(s2, _mm256_andnot_si256(s3, s4));
            s[j+3] = _mm256_xor_si256(s3, _mm256_andnot_si256(s4, s0));
            s[j+4] = _mm256_xor_si256(s4, _mm256_andnot_si256(s0, s1));
        }
        s[0] = _mm256_xor_si256(s[0], _mm256_set1_epi64x(KeccakF_RoundConstants[r]));
    }
}

void shake128x4_absorb_once(Keccak4x_State *state, const uint8_t *in0, const uint8_t *in1, const uint8_t *in2, const uint8_t *in3, size_t inlen) {
    memset(state, 0, sizeof(Keccak4x_State));
    uint64_t *s64 = (uint64_t*)state->s;
    for (size_t i = 0; i < inlen; i++) {
        int w = i / 8; int b = i % 8;
        s64[w*4 + 0] ^= (uint64_t)in0[i] << (8*b);
        s64[w*4 + 1] ^= (uint64_t)in1[i] << (8*b);
        s64[w*4 + 2] ^= (uint64_t)in2[i] << (8*b);
        s64[w*4 + 3] ^= (uint64_t)in3[i] << (8*b);
    }
    int w = inlen / 8; int b = inlen % 8;
    s64[w*4+0] ^= 0x1FULL << (8*b); s64[w*4+1] ^= 0x1FULL << (8*b);
    s64[w*4+2] ^= 0x1FULL << (8*b); s64[w*4+3] ^= 0x1FULL << (8*b);
    s64[20*4+0] ^= 0x80ULL << 56; s64[20*4+1] ^= 0x80ULL << 56;
    s64[20*4+2] ^= 0x80ULL << 56; s64[20*4+3] ^= 0x80ULL << 56;
}

void shake128x4_squeezeblocks(uint8_t *out0, uint8_t *out1, uint8_t *out2, uint8_t *out3, size_t nblocks, Keccak4x_State *state) {
    uint64_t *s64 = (uint64_t*)state->s;
    while(nblocks > 0) {
        KeccakP1600_times4_AVX2(state->s);
        for(int i=0; i<168; ++i) {
            int w = i/8; int b = i%8;
            out0[i] = (s64[w*4+0] >> (8*b)) & 0xFF;
            out1[i] = (s64[w*4+1] >> (8*b)) & 0xFF;
            out2[i] = (s64[w*4+2] >> (8*b)) & 0xFF;
            out3[i] = (s64[w*4+3] >> (8*b)) & 0xFF;
        }
        out0 += 168; out1 += 168; out2 += 168; out3 += 168;
        nblocks--;
    }
}