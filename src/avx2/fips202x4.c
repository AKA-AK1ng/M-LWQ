#include <stddef.h>
#include <stdint.h>
#include <immintrin.h>
#include "fips202x4.h"

// 补全常量定义
#define SHAKE128_RATE 168
#define SHAKE256_RATE 136
#define SHA3_256_RATE 136

// [FIX] 移除 cdecl 宏，直接声明函数
// 对应 src/avx2/keccak4x.c 中的定义
extern void KeccakP1600times4_PermuteAll_24rounds(__m256i *s);

void keccakx4_absorb_once(
    __m256i *s,
    unsigned int r,
    const uint8_t *in0,
    const uint8_t *in1,
    const uint8_t *in2,
    const uint8_t *in3,
    size_t inlen,
    uint8_t p
) {
    size_t i;
    uint64_t pos = 0;
    __m256i t, idx;

    // Initialize state
    for(i=0; i<25; i++)
        s[i] = _mm256_setzero_si256();

    // Prepare gather index
    idx = _mm256_set_epi64x(
        (long long)in3, (long long)in2, (long long)in1, (long long)in0
    );

    while(inlen >= r) {
        for(i=0; i<r/8; i++) {
            // Gather 64-bit word from each input stream
            t = _mm256_i64gather_epi64((long long *)pos, idx, 1);
            s[i] ^= t;
        }
        
        // [FIX] 直接调用，不使用 cdecl 宏
        KeccakP1600times4_PermuteAll_24rounds(s);
        
        inlen -= r;
        pos += r;
    }

    // Process remaining full 8-byte blocks
    for(i=0; i<inlen/8; i++) {
        t = _mm256_i64gather_epi64((long long *)pos, idx, 1);
        s[i] ^= t;
    }
    
    // Padding Logic (Cleaned up unused variables)
    const uint8_t *ins[4] = {in0+pos, in1+pos, in2+pos, in3+pos};
    size_t rem = inlen % 8; // Remaining bytes

    for(int stream=0; stream<4; stream++) {
        // 1. Handle remaining bytes (0-7 bytes)
        for(i=0; i<rem; i++) {
            uint64_t lane_idx = (i + inlen - rem) / 8;
            int byte_in_lane = (i + inlen - rem) % 8;
            uint64_t val = (uint64_t)ins[stream][i];
            ((uint64_t*)&s[lane_idx])[stream] ^= (val << (8*byte_in_lane));
        }
        
        // 2. Domain separator p
        uint64_t lane_idx = inlen / 8;
        int byte_in_lane = inlen % 8;
        ((uint64_t*)&s[lane_idx])[stream] ^= ((uint64_t)p << (8*byte_in_lane));
        
        // 3. Final bit 0x80 at rate-1
        lane_idx = (r - 1) / 8;
        byte_in_lane = (r - 1) % 8;
        ((uint64_t*)&s[lane_idx])[stream] ^= (0x80ULL << (8*byte_in_lane));
    }
    
    KeccakP1600times4_PermuteAll_24rounds(s);
}

void keccakx4_squeezeblocks(
    uint8_t *out0,
    uint8_t *out1,
    uint8_t *out2,
    uint8_t *out3,
    size_t nblocks,
    unsigned int r,
    __m256i *s
) {
    size_t i;
    __m256i t;
    
    // [FIX] 移除了未使用的 idx 变量

    while(nblocks > 0) {
        for(i=0; i<r/8; i++) {
            t = s[i];
            // Manually scatter the 4 lanes of t into the 4 output buffers
            // This avoids AVX512 scatter instructions
            uint64_t *v = (uint64_t*)&t;
            ((uint64_t*)out0)[i] = v[0];
            ((uint64_t*)out1)[i] = v[1];
            ((uint64_t*)out2)[i] = v[2];
            ((uint64_t*)out3)[i] = v[3];
        }
        
        KeccakP1600times4_PermuteAll_24rounds(s);
        
        out0 += r;
        out1 += r;
        out2 += r;
        out3 += r;
        nblocks--;
    }
}

void shake128x4_absorb_once(
    keccakx4_state *state,
    const uint8_t *in0,
    const uint8_t *in1,
    const uint8_t *in2,
    const uint8_t *in3,
    size_t inlen
) {
    keccakx4_absorb_once(state->s, SHAKE128_RATE, in0, in1, in2, in3, inlen, 0x1F);
}

void shake128x4_squeezeblocks(
    uint8_t *out0,
    uint8_t *out1,
    uint8_t *out2,
    uint8_t *out3,
    size_t nblocks,
    keccakx4_state *state
) {
    keccakx4_squeezeblocks(out0, out1, out2, out3, nblocks, SHAKE128_RATE, state->s);
}

void shake256x4_absorb_once(
    keccakx4_state *state,
    const uint8_t *in0,
    const uint8_t *in1,
    const uint8_t *in2,
    const uint8_t *in3,
    size_t inlen
) {
    keccakx4_absorb_once(state->s, SHAKE256_RATE, in0, in1, in2, in3, inlen, 0x1F);
}

void shake256x4_squeezeblocks(
    uint8_t *out0,
    uint8_t *out1,
    uint8_t *out2,
    uint8_t *out3,
    size_t nblocks,
    keccakx4_state *state
) {
    keccakx4_squeezeblocks(out0, out1, out2, out3, nblocks, SHAKE256_RATE, state->s);
}