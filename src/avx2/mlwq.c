#include "mlwq.h"
#include "poly.h"
#include "ntt.h"
#include "xof.h"
#include "../common/random.h"
#include "../common/fips202.h" 
#include <string.h>
#include <immintrin.h> // 必须包含 AVX2 头文件

// 引入并行哈希头文件
#include "fips202x4.h"

// 外部函数声明
extern void avx_poly_tobytes(uint8_t *r, const poly *a);
extern void avx_poly_frombytes(poly *r, const uint8_t *a);
extern void avx_poly_compress_u(uint8_t *r, const poly *a);
extern void avx_poly_decompress_u(poly *r, const uint8_t *a);
extern void avx_poly_compress_v(uint8_t *r, const poly *a);
extern void avx_poly_decompress_v(poly *r, const uint8_t *a);

// =========================================================================
// AVX2 SIMD 优化的 CBD3
// =========================================================================

static void avx_cbd3_simd(poly *r, const uint8_t *buf) {
    // 预计算常量
    const __m256i mask_249249 = _mm256_set1_epi32(0x00249249);
    const __m256i mask_7      = _mm256_set1_epi32(0x7);

    const __m256i shuf_mask = _mm256_setr_epi8(
        0, 1, 2, -1,  3, 4, 5, -1,  6, 7, 8, -1,  9, 10, 11, -1, // Low 128
        0, 1, 2, -1,  3, 4, 5, -1,  6, 7, 8, -1,  9, 10, 11, -1  // High 128 (reuse logic)
    );
    
    for (int i = 0; i < MLWQ_N / 16; i++) {
        
        __m128i lo = _mm_loadu_si128((const __m128i *)(buf));      // Load 16 bytes (use first 12)
        __m128i hi = _mm_loadu_si128((const __m128i *)(buf + 12)); // Load next 16 bytes (use first 12)
        __m256i t_vec = _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);
        
        buf += 24; // Advance input pointer
        
        // 2. 解包 (3 bytes -> 32-bit int)
        t_vec = _mm256_shuffle_epi8(t_vec, shuf_mask);
        
        // 3. 位运算 (并行计算 8 组 CBD)
        __m256i d = _mm256_and_si256(t_vec, mask_249249);
        d = _mm256_add_epi32(d, _mm256_and_si256(_mm256_srli_epi32(t_vec, 1), mask_249249));
        d = _mm256_add_epi32(d, _mm256_and_si256(_mm256_srli_epi32(t_vec, 2), mask_249249));
        uint32_t temp[8];
        _mm256_storeu_si256((__m256i*)temp, d);
        
        for(int k=0; k<8; k++) {
            uint32_t val = temp[k];
            int16_t *out_ptr = r->coeffs + i*32 + k*4;
            
            // 展开 4 个系数
            out_ptr[0] = (int16_t)((val >> 0) & 0x7) - (int16_t)((val >> 3) & 0x7);
            out_ptr[1] = (int16_t)((val >> 6) & 0x7) - (int16_t)((val >> 9) & 0x7);
            out_ptr[2] = (int16_t)((val >> 12) & 0x7) - (int16_t)((val >> 15) & 0x7);
            out_ptr[3] = (int16_t)((val >> 18) & 0x7) - (int16_t)((val >> 21) & 0x7);
        }
    }
}

// -------------------------------------------------------------------------
// PKE KeyGen
// -------------------------------------------------------------------------
void avx_mlwq_keygen(mlwq_pk *pk, mlwq_sk *sk, const uint8_t *seed_A, const uint8_t *seed_d) {
    // 1. 生成矩阵 A (SHAKE128)
    poly_matrix A;
    avx_xof_expand_matrix(&A, seed_A);
    
    // 2. 并行采样私钥 s (SHAKE256x4 + AVX CBD3)
    uint8_t seed_s[32];
    random_bytes(seed_s, 32); 

    uint8_t seeds[4][33];
    const uint8_t *in_ptrs[4];
    
    for(int i=0; i<4; i++) {
        memcpy(seeds[i], seed_s, 32);
        seeds[i][32] = i; 
        in_ptrs[i] = seeds[i];
    }
    
    // CBD3 需要 192 bytes, SHAKE256 rate 136. Need 2 blocks.
    uint8_t out[4][136 * 2]; 
    
    keccakx4_state state;
    shake256x4_absorb_once(&state, in_ptrs[0], in_ptrs[1], in_ptrs[2], in_ptrs[3], 33);
    shake256x4_squeezeblocks(out[0], out[1], out[2], out[3], 2, &state);
    
    for(int i=0; i<MLWQ_K; i++) {
        // [Change] 使用 AVX2 优化的 CBD3
        avx_cbd3_simd(&sk->s.vec[i], out[i]);
    }

    // 3. 生成 d_pk (SHAKE128)
    poly_vec d_pk;
    avx_xof_expand_poly_vec(&d_pk, seed_d, MLWQ_Q / P_PK);
    
    // 4. 计算 As + d
    poly_vec As;
    avx_poly_matrix_vec_mul(&As, &A, &sk->s);
    
    memcpy(pk->seed_A, seed_A, 32);
    memcpy(pk->seed_d, seed_d, 32);
    
    for(int i=0; i<MLWQ_K; ++i)
        avx_poly_quantize(&pk->b_q.vec[i], &As.vec[i], &d_pk.vec[i], P_PK);
}

// -------------------------------------------------------------------------
// PKE Encrypt
// -------------------------------------------------------------------------
void avx_mlwq_encrypt(mlwq_ciphertext *ct, const mlwq_pk *pk, const uint8_t *msg, const uint8_t *seed_ct) {
    poly_matrix A;
    avx_xof_expand_matrix(&A, pk->seed_A);
    
    // 1. 并行采样噪声 r (SHAKE256x4 + AVX CBD3)
    poly_vec r;
    
    uint8_t seeds[4][33];
    const uint8_t *in_ptrs[4];
    uint8_t out[4][136 * 2]; 
    
    for(int i=0; i<4; i++) {
        memcpy(seeds[i], seed_ct, 32);
        seeds[i][32] = i; 
        in_ptrs[i] = seeds[i];
    }
    
    keccakx4_state state;
    shake256x4_absorb_once(&state, in_ptrs[0], in_ptrs[1], in_ptrs[2], in_ptrs[3], 33);
    shake256x4_squeezeblocks(out[0], out[1], out[2], out[3], 2, &state);
    
    for(int i=0; i<MLWQ_K; i++) {
        // [Change] 使用 AVX2 优化的 CBD3
        avx_cbd3_simd(&r.vec[i], out[i]);
    }
    
    // 2. 生成 d_u (SHAKE128, XOF)
    poly_vec d_u;
    poly d_v;
    
    uint8_t d_seed[33];
    memcpy(d_seed, seed_ct, 32); 
    d_seed[32] = MLWQ_K; 
    avx_xof_expand_poly_vec(&d_u, d_seed, MLWQ_Q / P_U);
    
    // 3. 生成 d_v (SHAKE128, XOF)
    d_seed[32] = MLWQ_K + 1; 
    uint8_t buf_v[MLWQ_N*2];
    shake128(buf_v, sizeof(buf_v), d_seed, 33);
    
    int32_t mod_v = MLWQ_Q / P_V;
    for(int k=0; k<MLWQ_N; ++k) {
        uint16_t val = (uint16_t)buf_v[2*k] | ((uint16_t)buf_v[2*k+1]<<8);
        d_v.coeffs[k] = val % mod_v;
    }

    // 后续计算...
    poly_matrix At;
    for(int i=0; i<MLWQ_K; ++i) for(int j=0; j<MLWQ_K; ++j) At.row[i].vec[j] = A.row[j].vec[i];
    
    poly_vec Atr;
    avx_poly_matrix_vec_mul(&Atr, &At, &r);
    
    poly_vec b_deq;
    for(int i=0; i<MLWQ_K; ++i) avx_poly_dequantize(&b_deq.vec[i], &pk->b_q.vec[i], P_PK);
    
    poly v_val;
    avx_poly_vec_transpose_mul(&v_val, &b_deq, &r);
    
    poly m_poly;
    avx_poly_msg_encode(&m_poly, msg);
    
    poly v_final;
    avx_poly_add(&v_final, &v_val, &m_poly);
    
    for(int i=0; i<MLWQ_K; ++i) avx_poly_quantize(&ct->u.vec[i], &Atr.vec[i], &d_u.vec[i], P_U);
    avx_poly_quantize(&ct->v, &v_final, &d_v, P_V);
}

// -------------------------------------------------------------------------
// Decrypt / KEM Wrappers (保持不变)
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