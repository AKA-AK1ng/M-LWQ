#include <stdint.h>
#include <string.h>
#include <immintrin.h>
#include "xof.h"
#include "fips202x4.h"
#include "../common/fips202.h"
#include "params.h"

#define MLWQ_Q 3329
#define REJ_UNIFORM_NBLOCKS ((12 * MLWQ_N / 8 * (1 << 12) / MLWQ_Q + SHAKE128_RATE) / SHAKE128_RATE)

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
    
    // 缓冲区大小：4路 * REJ_UNIFORM_NBLOCKS * SHAKE128_RATE bytes
    uint8_t out[4][REJ_UNIFORM_NBLOCKS * SHAKE128_RATE] __attribute__((aligned(32)));
    uint8_t more[4][SHAKE128_RATE] __attribute__((aligned(32)));
    
    while (batch_idx < total_polys) {
        // 1. 确定当前批次处理多少个 (1~4)
        unsigned int remain = total_polys - batch_idx;
        unsigned int count = (remain >= 4) ? 4 : remain;
        
        // 2. 准备种子
        for(unsigned int k=0; k<count; k++) {
            unsigned int current = batch_idx + k;
            unsigned int r = current / MLWQ_K; // 行索引
            unsigned int c = current % MLWQ_K; // 列索引
            
            memcpy(seeds[k], seed, 32);
            // Kyber/MLWQ 标准序: seed || j || i  (Col, Row)
            // 注意：Little Endian 下，先行后列遍历对应 Nonce 顺序
            seeds[k][32] = c; 
            seeds[k][33] = r; 
            in_ptrs[k] = seeds[k];
        }
        
        // 如果不足4个，剩下的指针指向 seeds[0] 以防 crash (AVX Load 需要有效地址)
        for(unsigned int k=count; k<4; k++) {
            in_ptrs[k] = seeds[0];
        }
        
        // 3. 运行 4x SHAKE
        keccakx4_state state;
        shake128x4_absorb_once(&state, in_ptrs[0], in_ptrs[1], in_ptrs[2], in_ptrs[3], 34);
        shake128x4_squeezeblocks(out[0], out[1], out[2], out[3], REJ_UNIFORM_NBLOCKS, &state);
        
        // 4. 解析输出
        unsigned int ctr[4] = {0, 0, 0, 0};
        unsigned int pos[4] = {0, 0, 0, 0};
        unsigned int max_len = REJ_UNIFORM_NBLOCKS * SHAKE128_RATE;
        for(unsigned int k=0; k<count; k++) {
            unsigned int current = batch_idx + k;
            unsigned int r = current / MLWQ_K;
            unsigned int c = current % MLWQ_K;
            
            int16_t *poly_r = A->row[r].vec[c].coeffs;
            uint8_t *buf = out[k];
            
            ctr[k] = rej_uniform_avx2(poly_r, buf, max_len, ctr[k], &pos[k]);
        }

        for (unsigned int k = count; k < 4; k++) {
            ctr[k] = MLWQ_N;
        }

        while (ctr[0] < MLWQ_N || ctr[1] < MLWQ_N || ctr[2] < MLWQ_N || ctr[3] < MLWQ_N) {
            shake128x4_squeezeblocks(more[0], more[1], more[2], more[3], 1, &state);

            for (unsigned int k = 0; k < count; k++) {
                unsigned int current = batch_idx + k;
                unsigned int r = current / MLWQ_K;
                unsigned int c = current % MLWQ_K;
                int16_t *poly_r = A->row[r].vec[c].coeffs;
                unsigned int local_pos = 0;
                ctr[k] = rej_uniform_avx2(poly_r, more[k], SHAKE128_RATE, ctr[k], &local_pos);
            }
        }
        
        batch_idx += count;
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
                v->vec[current_idx].coeffs[j] = val % modulus; 
            }
        }
        
        batch_idx += count;
    }
}
