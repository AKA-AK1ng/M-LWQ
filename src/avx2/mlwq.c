#include "mlwq.h"
#include "poly.h"
#include "ntt.h"
#include "xof.h"
#include "../common/random.h"
#include "../common/fips202.h" 
#include <string.h>

// [关键优化] 引入并行哈希头文件
// 请确保 fips202x4.c 和汇编文件已被加入编译列表
#include "fips202x4.h"

// 声明外部的 AVX CBD2 函数 (如果你将其放在了 cbd.c 中)
// 如果你没有单独的 cbd.c，也可以将那个函数复制到这里变成 static 函数
extern void avx_poly_cbd2(poly *r, const uint8_t buf[128]);

// -------------------------------------------------------------------------
// PKE KeyGen (Parallel Optimized)
// -------------------------------------------------------------------------
void avx_mlwq_keygen(mlwq_pk *pk, mlwq_sk *sk, const uint8_t *seed_A, const uint8_t *seed_d) {
    // 1. 生成矩阵 A
    // avx_xof_expand_matrix 内部通常已经包含 4路并行优化
    poly_matrix A;
    avx_xof_expand_matrix(&A, seed_A);
    
    // 2. [优化] 并行采样私钥 s
    // 我们需要生成 K=2 个多项式。使用 shake128x4 一次性生成。
    
    // 为了驱动 SHAKE，我们需要一个随机种子
    uint8_t seed_s[32];
    random_bytes(seed_s, 32); 

    uint8_t seeds[4][33];
    const uint8_t *in_ptrs[4];
    
    // 准备 4 路输入: seed + nonce(0,1,2,3)
    for(int i=0; i<4; i++) {
        memcpy(seeds[i], seed_s, 32);
        seeds[i][32] = i; 
        in_ptrs[i] = seeds[i];
    }
    
    // CBD2 只需要 128 字节 (256 * 4bit)
    // SHAKE128 一个 block 是 168 字节，足够了
    uint8_t out[4][168]; 
    
    keccakx4_state state;
    shake128x4_absorb_once(&state, in_ptrs[0], in_ptrs[1], in_ptrs[2], in_ptrs[3], 33);
    shake128x4_squeezeblocks(out[0], out[1], out[2], out[3], 1, &state);
    
    // 解析输出 (只用到前 K 个流)
    for(int i=0; i<MLWQ_K; i++) {
        avx_poly_cbd2(&sk->s.vec[i], out[i]);
    }

    // 3. 生成抖动 (Dither)
    // 这里保持标量调用即可，或者也可以并行化，但需要不同的模数处理逻辑
    poly_vec d_pk;
    avx_xof_expand_poly_vec(&d_pk, seed_d, MLWQ_Q / P_PK);
    
    // 4. 矩阵乘法 A * s
    poly_vec As;
    avx_poly_matrix_vec_mul(&As, &A, &sk->s);
    
    // 5. 量化 b = Quantize(A*s + d)
    memcpy(pk->seed_A, seed_A, 32);
    memcpy(pk->seed_d, seed_d, 32);
    
    for(int i=0; i<MLWQ_K; ++i)
        avx_poly_quantize(&pk->b_q.vec[i], &As.vec[i], &d_pk.vec[i], P_PK);
}

// -------------------------------------------------------------------------
// PKE Encrypt (Parallel Optimized)
// -------------------------------------------------------------------------
void avx_mlwq_encrypt(mlwq_ciphertext *ct, const mlwq_pk *pk, const uint8_t *msg, const uint8_t *seed_ct) {
    // 1. 生成矩阵 A
    poly_matrix A;
    avx_xof_expand_matrix(&A, pk->seed_A);
    
    // 2. [优化] 并行采样噪声 r
    // 原理同 KeyGen，利用 seed_ct 和 4x SHAKE
    poly_vec r;
    
    uint8_t seeds[4][33];
    const uint8_t *in_ptrs[4];
    uint8_t out[4][168];
    
    for(int i=0; i<4; i++) {
        memcpy(seeds[i], seed_ct, 32);
        seeds[i][32] = i; // Nonce 0, 1 ...
        in_ptrs[i] = seeds[i];
    }
    
    keccakx4_state state;
    shake128x4_absorb_once(&state, in_ptrs[0], in_ptrs[1], in_ptrs[2], in_ptrs[3], 33);
    shake128x4_squeezeblocks(out[0], out[1], out[2], out[3], 1, &state);
    
    for(int i=0; i<MLWQ_K; i++) {
        avx_poly_cbd2(&r.vec[i], out[i]);
    }
    
    // 3. 生成加密所需的抖动 (d_u, d_v)
    poly_vec d_u;
    poly d_v;
    
    // d_u 使用 Nonce = K
    uint8_t d_seed[33];
    memcpy(d_seed, seed_ct, 32); 
    d_seed[32] = MLWQ_K; 
    avx_xof_expand_poly_vec(&d_u, d_seed, MLWQ_Q / P_U);
    
    // d_v 使用 Nonce = K + 1
    // 这里使用标量 SHAKE 即可，因为只是 1 个多项式
    d_seed[32] = MLWQ_K + 1;
    uint8_t buf_v[MLWQ_N*2];
    shake128(buf_v, sizeof(buf_v), d_seed, 33);
    
    int32_t mod_v = MLWQ_Q / P_V;
    // 解析 d_v (简单的均匀模分布)
    for(int k=0; k<MLWQ_N; ++k) {
        uint16_t val = (uint16_t)buf_v[2*k] | ((uint16_t)buf_v[2*k+1]<<8);
        d_v.coeffs[k] = val % mod_v;
    }

    // 4. 计算 u = A^T * r
    poly_matrix At;
    for(int i=0; i<MLWQ_K; ++i) for(int j=0; j<MLWQ_K; ++j) At.row[i].vec[j] = A.row[j].vec[i];
    
    poly_vec Atr;
    avx_poly_matrix_vec_mul(&Atr, &At, &r);
    
    // 5. 计算 v = b^T * r + msg
    poly_vec b_deq;
    for(int i=0; i<MLWQ_K; ++i) avx_poly_dequantize(&b_deq.vec[i], &pk->b_q.vec[i], P_PK);
    
    poly v_val;
    avx_poly_vec_transpose_mul(&v_val, &b_deq, &r);
    
    poly m_poly;
    avx_poly_msg_encode(&m_poly, msg);
    
    poly v_final;
    avx_poly_add(&v_final, &v_val, &m_poly);
    
    // 6. 量化
    for(int i=0; i<MLWQ_K; ++i) avx_poly_quantize(&ct->u.vec[i], &Atr.vec[i], &d_u.vec[i], P_U);
    avx_poly_quantize(&ct->v, &v_final, &d_v, P_V);
}

// -------------------------------------------------------------------------
// PKE Decrypt (Arith Optimized)
// -------------------------------------------------------------------------
void avx_mlwq_decrypt(uint8_t *msg, const mlwq_sk *sk, const mlwq_ciphertext *ct) {
    poly_vec u_deq;
    for(int i=0; i<MLWQ_K; ++i) avx_poly_dequantize(&u_deq.vec[i], &ct->u.vec[i], P_U);
    
    poly v_deq;
    avx_poly_dequantize(&v_deq, &ct->v, P_V);
    
    poly s_t_u;
    avx_poly_vec_transpose_mul(&s_t_u, &sk->s, &u_deq);
    
    poly diff;
    avx_poly_sub(&diff, &v_deq, &s_t_u);
    
    avx_poly_msg_decode(msg, &diff);
}

// -------------------------------------------------------------------------
// KEM Wrappers
// -------------------------------------------------------------------------
void avx_mlwq_kem_keygen(mlwq_pk *pk, mlwq_kem_sk *sk) {
    uint8_t seed_A[32], seed_d[32];
    random_bytes(seed_A, 32);
    random_bytes(seed_d, 32);
    
    avx_mlwq_keygen(pk, &sk->pke_sk, seed_A, seed_d);
    sk->pk = *pk;
    
    shake128(sk->h_pk, 32, (uint8_t*)pk, sizeof(mlwq_pk));
    random_bytes(sk->z, 32);
}

void avx_mlwq_kem_encaps(mlwq_ciphertext *ct, uint8_t *ss, const mlwq_pk *pk) {
    uint8_t m[32];
    random_bytes(m, 32);
    
    uint8_t buf[64];
    memcpy(buf, m, 32);
    shake128(buf+32, 32, (uint8_t*)pk, sizeof(mlwq_pk)); 
    
    uint8_t kr[64];
    shake128(kr, 64, buf, 64);
    
    memcpy(ss, kr, 32);
    avx_mlwq_encrypt(ct, pk, m, kr+32);
}

int avx_mlwq_kem_decaps(uint8_t *ss, const mlwq_kem_sk *sk, const mlwq_ciphertext *ct) {
    uint8_t m[32];
    avx_mlwq_decrypt(m, &sk->pke_sk, ct);
    
    uint8_t buf[64];
    memcpy(buf, m, 32);
    memcpy(buf+32, sk->h_pk, 32);
    
    uint8_t kr[64];
    shake128(kr, 64, buf, 64);
    
    mlwq_ciphertext ct_prime;
    avx_mlwq_encrypt(&ct_prime, &sk->pk, m, kr+32);
    
    if(memcmp(ct, &ct_prime, sizeof(mlwq_ciphertext)) == 0) {
        memcpy(ss, kr, 32);
        return 1;
    } else {
        shake128(ss, 32, sk->z, 32);
        return 0;
    }
}