#include <stdint.h>
#include <string.h>
#include "xof.h"
#include "../avx2/fips202x4.h"
#include "../avx2/rejsample.h"
#include "../common/fips202.h"
#include "../common/params.h"

static inline uint16_t fast_mod_u16(uint16_t val, uint16_t mod) {
    uint32_t recip = (uint32_t)(((uint64_t)1 << 32) / mod);
    uint32_t q = (uint32_t)(((uint64_t)val * recip) >> 32);
    uint32_t r = val - q * mod;
    if (r >= mod) r -= mod;
    return (uint16_t)r;
}

static inline uint16_t fast_mod_u16_precomp(uint16_t val, uint16_t mod, uint32_t recip) {
    uint32_t q = (uint32_t)(((uint64_t)val * recip) >> 32);
    uint32_t r = val - q * mod;
    if (r >= mod) r -= mod;
    return (uint16_t)r;
}

static inline uint16_t reduce_u16(uint16_t val, uint16_t mod, uint32_t recip) {
    if ((mod & (mod - 1)) == 0) {
        return (uint16_t)(val & (mod - 1));
    }
    return fast_mod_u16_precomp(val, mod, recip);
}

static unsigned int rej_uniform_avx2(int16_t *r,
                                     const uint8_t *buf,
                                     unsigned int buflen,
                                     unsigned int ctr,
                                     unsigned int *pos) {
    unsigned int pos_local = *pos;

    while (ctr < MLWQ_N && pos_local + 2 < buflen) {
        uint16_t val0 = ((uint16_t)buf[pos_local] | ((uint16_t)buf[pos_local + 1] << 8)) & 0xFFF;
        uint16_t val1 = ((uint16_t)(buf[pos_local + 1] >> 4) | ((uint16_t)buf[pos_local + 2] << 4)) & 0xFFF;
        pos_local += 3;

        if (val0 < MLWQ_Q) {
            r[ctr++] = (int16_t)val0;
        }
        if (ctr < MLWQ_N && val1 < MLWQ_Q) {
            r[ctr++] = (int16_t)val1;
        }
    }

    *pos = pos_local;
    return ctr;
}

static void shake128x4_init_and_squeeze(keccakx4_state *state,
                                        const uint8_t *in0,
                                        const uint8_t *in1,
                                        const uint8_t *in2,
                                        const uint8_t *in3,
                                        unsigned int inlen,
                                        uint8_t *out0,
                                        uint8_t *out1,
                                        uint8_t *out2,
                                        uint8_t *out3,
                                        unsigned int nblocks) {
    shake128x4_absorb_once(state, in0, in1, in2, in3, inlen);
    shake128x4_squeezeblocks(out0, out1, out2, out3, nblocks, state);
}

static void complete_rejection_x4(keccakx4_state *state,
                                  int16_t *poly_ptrs[4],
                                  unsigned int ctr[4]) {
    uint8_t more[4][SHAKE128_RATE] __attribute__((aligned(32)));
    unsigned int active_lanes = 4;
    int pending = 1;

    for (unsigned int k = 0; k < active_lanes; k++) {
        if (poly_ptrs[k] == NULL) {
            ctr[k] = MLWQ_N;
        }
    }

    while (pending) {
        pending = 0;
        for (unsigned int k = 0; k < active_lanes; k++) {
            if (ctr[k] < MLWQ_N) {
                pending = 1;
                break;
            }
        }
        if (!pending) {
            break;
        }
        shake128x4_squeezeblocks(more[0], more[1], more[2], more[3], 1, state);

        for (unsigned int k = 0; k < active_lanes; k++) {
            unsigned int local_pos = 0;
            if (ctr[k] < MLWQ_N && poly_ptrs[k] != NULL) {
                ctr[k] = rej_uniform_avx2(poly_ptrs[k], more[k], SHAKE128_RATE, ctr[k], &local_pos);
            }
        }
    }
}

// =========================================================================
// 1. 矩阵生成 (8-way 批处理，通过两组 4x SHAKE 实现)
// =========================================================================
void avx512_xof_expand_matrix(poly_matrix *A, const uint8_t *seed) {
    unsigned int total_polys = MLWQ_K * MLWQ_K;
    unsigned int batch_idx = 0;
    unsigned int row = batch_idx / MLWQ_K;
    unsigned int col = batch_idx % MLWQ_K;

    uint8_t seeds[8][34];
    const uint8_t *in_ptrs[8];
    uint8_t out[8][REJ_UNIFORM_AVX_BUFLEN] __attribute__((aligned(32)));

    while (batch_idx < total_polys) {
        unsigned int remain = total_polys - batch_idx;
        unsigned int count = (remain >= 8) ? 8 : remain;

        unsigned int r = row;
        unsigned int c = col;
        for (unsigned int k = 0; k < count; k++) {
            memcpy(seeds[k], seed, 32);
            seeds[k][32] = c;
            seeds[k][33] = r;
            in_ptrs[k] = seeds[k];

            c++;
            if (c == MLWQ_K) {
                c = 0;
                r++;
            }
        }
        for (unsigned int k = count; k < 8; k++) {
            in_ptrs[k] = seeds[0];
        }

        keccakx4_state state0;
        keccakx4_state state1;

        shake128x4_init_and_squeeze(&state0,
                                    in_ptrs[0], in_ptrs[1], in_ptrs[2], in_ptrs[3], 34,
                                    out[0], out[1], out[2], out[3], REJ_UNIFORM_AVX_NBLOCKS);
        shake128x4_init_and_squeeze(&state1,
                                    in_ptrs[4], in_ptrs[5], in_ptrs[6], in_ptrs[7], 34,
                                    out[4], out[5], out[6], out[7], REJ_UNIFORM_AVX_NBLOCKS);

        unsigned int ctr[8] = {0};
        int16_t *poly_ptrs[8] = {0};
        for (unsigned int k = 0; k < count; k++) {
            unsigned int current = batch_idx + k;
            unsigned int rr = current / MLWQ_K;
            unsigned int cc = current % MLWQ_K;
            poly_ptrs[k] = A->row[rr].vec[cc].coeffs;
            ctr[k] = rej_uniform_avx(poly_ptrs[k], out[k]);
        }

        int16_t *poly_ptrs0[4] = {poly_ptrs[0], poly_ptrs[1], poly_ptrs[2], poly_ptrs[3]};
        int16_t *poly_ptrs1[4] = {poly_ptrs[4], poly_ptrs[5], poly_ptrs[6], poly_ptrs[7]};
        unsigned int ctr0[4] = {ctr[0], ctr[1], ctr[2], ctr[3]};
        unsigned int ctr1[4] = {ctr[4], ctr[5], ctr[6], ctr[7]};

        complete_rejection_x4(&state0, poly_ptrs0, ctr0);
        complete_rejection_x4(&state1, poly_ptrs1, ctr1);

        batch_idx += count;
        row = r;
        col = c;
    }
}

// =========================================================================
// 2. 向量生成 (8-way 批处理，通过两组 4x SHAKE 实现)
// =========================================================================
void avx512_xof_expand_poly_vec(poly_vec *v, const uint8_t *seed, int32_t modulus) {
    const unsigned int nblocks = (MLWQ_N * 2 + SHAKE128_RATE - 1) / SHAKE128_RATE;
    unsigned int batch_idx = 0;
    uint8_t seeds[8][33];
    const uint8_t *in_ptrs[8];
    uint8_t out[8][SHAKE128_RATE * nblocks];
    uint16_t mod = (uint16_t)modulus;
    uint32_t recip = (uint32_t)(((uint64_t)1 << 32) / mod);

    while (batch_idx < MLWQ_K) {
        unsigned int remain = MLWQ_K - batch_idx;
        unsigned int count = (remain >= 8) ? 8 : remain;

        for (unsigned int k = 0; k < count; k++) {
            unsigned int current_idx = batch_idx + k;
            memcpy(seeds[k], seed, 32);
            seeds[k][32] = seed[32] + current_idx;
            in_ptrs[k] = seeds[k];
        }
        for (unsigned int k = count; k < 8; k++) {
            in_ptrs[k] = seeds[0];
        }

        keccakx4_state state0;
        keccakx4_state state1;
        shake128x4_init_and_squeeze(&state0,
                                    in_ptrs[0], in_ptrs[1], in_ptrs[2], in_ptrs[3], 33,
                                    out[0], out[1], out[2], out[3], nblocks);
        shake128x4_init_and_squeeze(&state1,
                                    in_ptrs[4], in_ptrs[5], in_ptrs[6], in_ptrs[7], 33,
                                    out[4], out[5], out[6], out[7], nblocks);

        for (unsigned int k = 0; k < count; k++) {
            unsigned int current_idx = batch_idx + k;
            uint8_t *buf = out[k];
            for (int j = 0; j < MLWQ_N; j++) {
                uint16_t val = (uint16_t)buf[2 * j] | ((uint16_t)buf[2 * j + 1] << 8);
                v->vec[current_idx].coeffs[j] = reduce_u16(val, mod, recip);
            }
        }

        batch_idx += count;
    }
}

// =========================================================================
// 3. 单多项式生成 (用于 d_v 等单项)
// =========================================================================
void avx512_xof_expand_poly(poly *v, const uint8_t *seed, int32_t modulus) {
    const unsigned int nblocks = (MLWQ_N * 2 + SHAKE128_RATE - 1) / SHAKE128_RATE;
    uint8_t out[SHAKE128_RATE * nblocks];
    uint16_t mod = (uint16_t)modulus;
    uint32_t recip = (uint32_t)(((uint64_t)1 << 32) / mod);

    keccak_state state;
    shake128_absorb_once(&state, seed, 33);
    shake128_squeezeblocks(out, nblocks, &state);

    uint8_t *buf = out;
    for (int j = 0; j < MLWQ_N; j++) {
        uint16_t val = (uint16_t)buf[2 * j] | ((uint16_t)buf[2 * j + 1] << 8);
        v->coeffs[j] = reduce_u16(val, mod, recip);
    }
}
