#include <stdint.h>
#include <string.h>
#include <immintrin.h>
#include "xof.h"
#include "fips202x4.h"
#include "rejsample.h"
#include "../common/fips202.h"
#include "params.h"

#define MLWQ_Q 3329

static inline uint16_t fast_mod_u16(uint16_t val, uint16_t mod) {
    uint32_t recip = (uint32_t)(((uint64_t)1 << 32) / mod);
    uint32_t q = (uint32_t)(((uint64_t)val * recip) >> 32);
    uint32_t r = val - q * mod;
    if (r >= mod) r -= mod;
    return (uint16_t)r;
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

// =========================================================================
// 1. 矩阵生成 (支持任意 K 的分批处理)
// =========================================================================
void avx_xof_expand_matrix(poly_matrix *A, const uint8_t *seed) {
    // 每次处理 4 个多项式
    // K=2 -> 总共 4 个 (1 batch)
    // K=3 -> 总共 9 个 (3 batches: 4, 4, 1)
    // K=4 -> 总共 16 个 (4 batches: 4, 4, 4, 4)
    
    unsigned int total_polys = MLWQ_K * MLWQ_K;
    unsigned int batch_idx = 0;
    
    uint8_t seeds[4][34];
    const uint8_t *in_ptrs[4];
    
    // 缓冲区大小：4路 * REJ_UNIFORM_AVX_BUFLEN bytes
    uint8_t out[4][REJ_UNIFORM_AVX_BUFLEN] __attribute__((aligned(32)));
    uint8_t more[4][SHAKE128_RATE] __attribute__((aligned(32)));
    
    __m256i seed_vec = _mm256_loadu_si256((const __m256i *)seed);
    unsigned int row = batch_idx / MLWQ_K;
    unsigned int col = batch_idx % MLWQ_K;

    while (batch_idx < total_polys) {
        // 1. 确定当前批次处理多少个 (1~4)
        unsigned int remain = total_polys - batch_idx;
        unsigned int count = (remain >= 4) ? 4 : remain;
        
        // 2. 准备种子
        unsigned int r = row;
        unsigned int c = col;
        for (unsigned int k = 0; k < count; k++) {
            _mm256_storeu_si256((__m256i *)seeds[k], seed_vec);
            // Kyber/MLWQ 标准序: seed || j || i  (Col, Row)
            seeds[k][32] = c;
            seeds[k][33] = r;
            in_ptrs[k] = seeds[k];

            c++;
            if (c == MLWQ_K) {
                c = 0;
                r++;
            }
        }
        
        // 如果不足4个，剩下的指针指向 seeds[0] 以防 crash (AVX Load 需要有效地址)
        for(unsigned int k=count; k<4; k++) {
            in_ptrs[k] = seeds[0];
        }
        
        // 3. 运行 4x SHAKE
        keccakx4_state state;
        shake128x4_absorb_once(&state, in_ptrs[0], in_ptrs[1], in_ptrs[2], in_ptrs[3], 34);
        shake128x4_squeezeblocks(out[0], out[1], out[2], out[3], REJ_UNIFORM_AVX_NBLOCKS, &state);
        
        // 4. 解析输出
        unsigned int ctr[4] = {0, 0, 0, 0};
        int16_t *poly_ptrs[4] = {0};
        for(unsigned int k=0; k<count; k++) {
            unsigned int current = batch_idx + k;
            unsigned int r = current / MLWQ_K;
            unsigned int c = current % MLWQ_K;
            poly_ptrs[k] = A->row[r].vec[c].coeffs;
            ctr[k] = rej_uniform_avx(poly_ptrs[k], out[k]);
        }

        for (unsigned int k = count; k < 4; k++) {
            ctr[k] = MLWQ_N;
        }

        while (1) {
            int pending = 0;
            for (unsigned int k = 0; k < count; k++) {
                if (ctr[k] < MLWQ_N) {
                    pending = 1;
                    break;
                }
            }
            if (!pending) {
                break;
            }
            shake128x4_squeezeblocks(more[0], more[1], more[2], more[3], 1, &state);

            for (unsigned int k = 0; k < count; k++) {
                unsigned int local_pos = 0;
                if (ctr[k] < MLWQ_N) {
                    ctr[k] = rej_uniform_avx2(poly_ptrs[k], more[k], SHAKE128_RATE, ctr[k], &local_pos);
                }
            }
        }
        
        batch_idx += count;
        row = r;
        col = c;
    }
}

// =========================================================================
// 2. 向量生成 (修复 Nonce 覆盖问题)
// =========================================================================
void avx_xof_expand_poly_vec(poly_vec *v, const uint8_t *seed, int32_t modulus) {
    // 同样需要分批处理，支持 K > 4 的情况 (虽然 L5 K=4 正好填满，但为了鲁棒性)
    
    unsigned int batch_idx = 0;
    uint8_t seeds[4][33];
    const uint8_t *in_ptrs[4];
    uint8_t out[4][168 * 4]; 
    uint16_t mod = (uint16_t)modulus;
    
    while (batch_idx < MLWQ_K) {
        unsigned int remain = MLWQ_K - batch_idx;
        unsigned int count = (remain >= 4) ? 4 : remain;
        
        for(unsigned int k=0; k<count; k++) {
            unsigned int current_idx = batch_idx + k;
            
            memcpy(seeds[k], seed, 32);
            
            // [关键修复] 不要直接 = k !
            // 应该 = base_nonce + k
            // mlwq.c 在生成 d_u 时，传入的 seed[32] 已经是 MLWQ_K 了
            // 所以我们要加上 seed[32] 的值
            seeds[k][32] = seed[32] + current_idx; 
            
            in_ptrs[k] = seeds[k];
        }
        
        for(unsigned int k=count; k<4; k++) in_ptrs[k] = seeds[0];
        
        keccakx4_state state;
        shake128x4_absorb_once(&state, in_ptrs[0], in_ptrs[1], in_ptrs[2], in_ptrs[3], 33);
        shake128x4_squeezeblocks(out[0], out[1], out[2], out[3], 4, &state);
        
        for(unsigned int k=0; k<count; k++) {
            unsigned int current_idx = batch_idx + k;
            uint8_t *buf = out[k];
            for(int j=0; j<MLWQ_N; j++) {
                uint16_t val = (uint16_t)buf[2*j] | ((uint16_t)buf[2*j+1]<<8);
                v->vec[current_idx].coeffs[j] = fast_mod_u16(val, mod);
            }
        }
        
        batch_idx += count;
    }
}

// =========================================================================
// 3. 单多项式生成 (用于 d_v 等单项)
// =========================================================================
void avx_xof_expand_poly(poly *v, const uint8_t *seed, int32_t modulus) {
    uint8_t seeds[4][33];
    const uint8_t *in_ptrs[4];
    const unsigned int nblocks = (MLWQ_N * 2 + SHAKE128_RATE - 1) / SHAKE128_RATE;
    uint8_t out[4][SHAKE128_RATE * nblocks];
    uint16_t mod = (uint16_t)modulus;

    memcpy(seeds[0], seed, 33);
    in_ptrs[0] = seeds[0];
    for (int k = 1; k < 4; k++) {
        memcpy(seeds[k], seed, 33);
        in_ptrs[k] = seeds[k];
    }

    keccakx4_state state;
    shake128x4_absorb_once(&state, in_ptrs[0], in_ptrs[1], in_ptrs[2], in_ptrs[3], 33);
    shake128x4_squeezeblocks(out[0], out[1], out[2], out[3], nblocks, &state);

    uint8_t *buf = out[0];
    for (int j = 0; j < MLWQ_N; j++) {
        uint16_t val = (uint16_t)buf[2 * j] | ((uint16_t)buf[2 * j + 1] << 8);
        v->coeffs[j] = fast_mod_u16(val, mod);
    }
}
