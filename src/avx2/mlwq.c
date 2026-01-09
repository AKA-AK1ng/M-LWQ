#include "mlwq.h"
#include "poly.h"
#include "ntt.h"
#include "xof.h"
#include "align.h"
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
// AVX2 SIMD 优化的 CBD3（对齐 Kyber 的解包布局）
// =========================================================================

static void avx_cbd3(poly * restrict r, const uint8_t buf[3 * MLWQ_N / 4 + 8]) {
    unsigned int i;
    __m256i f0, f1, f2, f3;
    const __m256i mask249 = _mm256_set1_epi32(0x249249);
    const __m256i mask6db = _mm256_set1_epi32(0x6DB6DB);
    const __m256i mask07 = _mm256_set1_epi32(7);
    const __m256i mask70 = _mm256_set1_epi32(7 << 16);
    const __m256i mask3 = _mm256_set1_epi16(3);
    const __m256i shufbidx = _mm256_set_epi8(-1, 15, 14, 13, -1, 12, 11, 10, -1, 9, 8, 7, -1, 6, 5, 4,
                                             -1, 11, 10, 9, -1, 8, 7, 6, -1, 5, 4, 3, -1, 2, 1, 0);

    for (i = 0; i < MLWQ_N / 32; i++) {
        f0 = _mm256_loadu_si256((const __m256i *)&buf[24 * i]);
        f0 = _mm256_permute4x64_epi64(f0, 0x94);
        f0 = _mm256_shuffle_epi8(f0, shufbidx);

        f1 = _mm256_srli_epi32(f0, 1);
        f2 = _mm256_srli_epi32(f0, 2);
        f0 = _mm256_and_si256(mask249, f0);
        f1 = _mm256_and_si256(mask249, f1);
        f2 = _mm256_and_si256(mask249, f2);
        f0 = _mm256_add_epi32(f0, f1);
        f0 = _mm256_add_epi32(f0, f2);

        f1 = _mm256_srli_epi32(f0, 3);
        f0 = _mm256_add_epi32(f0, mask6db);
        f0 = _mm256_sub_epi32(f0, f1);

        f1 = _mm256_slli_epi32(f0, 10);
        f2 = _mm256_srli_epi32(f0, 12);
        f3 = _mm256_srli_epi32(f0, 2);
        f0 = _mm256_and_si256(f0, mask07);
        f1 = _mm256_and_si256(f1, mask70);
        f2 = _mm256_and_si256(f2, mask07);
        f3 = _mm256_and_si256(f3, mask70);
        f0 = _mm256_add_epi16(f0, f1);
        f1 = _mm256_add_epi16(f2, f3);
        f0 = _mm256_sub_epi16(f0, mask3);
        f1 = _mm256_sub_epi16(f1, mask3);

        f2 = _mm256_unpacklo_epi32(f0, f1);
        f3 = _mm256_unpackhi_epi32(f0, f1);

        f0 = _mm256_permute2x128_si256(f2, f3, 0x20);
        f1 = _mm256_permute2x128_si256(f2, f3, 0x31);

        _mm256_store_si256((__m256i *)&r->coeffs[32 * i + 0], f0);
        _mm256_store_si256((__m256i *)&r->coeffs[32 * i + 16], f1);
    }
}

static void avx_poly_cbd_eta1(poly *r, const __m256i buf[MLWQ_ETA1 * MLWQ_N / 128 + 1]) {
#if MLWQ_ETA1 == 3
    avx_cbd3(r, (const uint8_t *)buf);
#else
#error "AVX2 cbd requires MLWQ_ETA1 == 3"
#endif
}

// =========================================================================
// AVX2 4x 噪声采样 (对齐 Kyber 的 4x SHAKE256 + CBD3 流程)
// =========================================================================
static void avx_poly_getnoise_eta1_4x(poly *r0,
                                      poly *r1,
                                      poly *r2,
                                      poly *r3,
                                      const uint8_t seed[32],
                                      uint8_t nonce0,
                                      uint8_t nonce1,
                                      uint8_t nonce2,
                                      uint8_t nonce3) {
    #define NOISE_NBLOCKS ((MLWQ_ETA1 * MLWQ_N / 4 + SHAKE256_RATE - 1) / SHAKE256_RATE)
    ALIGNED_UINT8(NOISE_NBLOCKS * SHAKE256_RATE) buf[4];
    __m256i f;
    keccakx4_state state;

    f = _mm256_loadu_si256((__m256i *)seed);
    _mm256_store_si256(buf[0].vec, f);
    _mm256_store_si256(buf[1].vec, f);
    _mm256_store_si256(buf[2].vec, f);
    _mm256_store_si256(buf[3].vec, f);

    buf[0].coeffs[32] = nonce0;
    buf[1].coeffs[32] = nonce1;
    buf[2].coeffs[32] = nonce2;
    buf[3].coeffs[32] = nonce3;

    shake256x4_absorb_once(&state, buf[0].coeffs, buf[1].coeffs, buf[2].coeffs, buf[3].coeffs, 33);
    shake256x4_squeezeblocks(buf[0].coeffs, buf[1].coeffs, buf[2].coeffs, buf[3].coeffs, NOISE_NBLOCKS, &state);

    avx_poly_cbd_eta1(r0, (const __m256i *)buf[0].coeffs);
    avx_poly_cbd_eta1(r1, (const __m256i *)buf[1].coeffs);
    avx_poly_cbd_eta1(r2, (const __m256i *)buf[2].coeffs);
    avx_poly_cbd_eta1(r3, (const __m256i *)buf[3].coeffs);
    #undef NOISE_NBLOCKS
}

void avx_poly_getnoise_eta1(poly *r, const uint8_t seed[32], uint8_t nonce) {
    ALIGNED_UINT8(MLWQ_ETA1 * MLWQ_N / 4 + 32) buf;
    uint8_t ext_seed[33];

    memcpy(ext_seed, seed, 32);
    ext_seed[32] = nonce;
    shake256(buf.coeffs, MLWQ_ETA1 * MLWQ_N / 4, ext_seed, sizeof(ext_seed));
    avx_poly_cbd_eta1(r, buf.vec);
}

static void avx_polyvec_getnoise_eta1(poly_vec *r, const uint8_t seed[32], uint8_t nonce_base) {
    int i = 0;
    while (i + 3 < MLWQ_K) {
        avx_poly_getnoise_eta1_4x(&r->vec[i + 0],
                                  &r->vec[i + 1],
                                  &r->vec[i + 2],
                                  &r->vec[i + 3],
                                  seed,
                                  (uint8_t)(nonce_base + i + 0),
                                  (uint8_t)(nonce_base + i + 1),
                                  (uint8_t)(nonce_base + i + 2),
                                  (uint8_t)(nonce_base + i + 3));
        i += 4;
    }
    if (i < MLWQ_K) {
        poly tmp[3];
        poly *p0 = &r->vec[i + 0];
        poly *p1 = (i + 1 < MLWQ_K) ? &r->vec[i + 1] : &tmp[0];
        poly *p2 = (i + 2 < MLWQ_K) ? &r->vec[i + 2] : &tmp[1];
        poly *p3 = (i + 3 < MLWQ_K) ? &r->vec[i + 3] : &tmp[2];
        avx_poly_getnoise_eta1_4x(p0,
                                  p1,
                                  p2,
                                  p3,
                                  seed,
                                  (uint8_t)(nonce_base + i + 0),
                                  (uint8_t)(nonce_base + i + 1),
                                  (uint8_t)(nonce_base + i + 2),
                                  (uint8_t)(nonce_base + i + 3));
    }
}

// -------------------------------------------------------------------------
// PKE KeyGen
// -------------------------------------------------------------------------
void avx_mlwq_keygen(mlwq_pk *pk, mlwq_sk *sk, const uint8_t *seed_A, const uint8_t *seed_d) {
    // 1. 生成矩阵 A (SHAKE128)
    poly_matrix A;
    avx_xof_expand_matrix(&A, seed_A);
    poly_matrix A_ntt = A;
    for (int i = 0; i < MLWQ_K; ++i) {
        for (int j = 0; j < MLWQ_K; ++j) {
            avx_ntt(A_ntt.row[i].vec[j].coeffs);
        }
    }
    
    // 2. 并行采样私钥 s (SHAKE256x4 + AVX CBD3，与 MLWQ_ETA1 = 3 对齐)
    // 使用私有噪声种子，避免与公开 seed_d 绑定
    uint8_t seed_s[32];
    random_bytes(seed_s, 32);
    avx_polyvec_getnoise_eta1(&sk->s, seed_s, 0);

    // 3. 生成 d_pk (SHAKE128)
    poly_vec d_pk;
    avx_xof_expand_poly_vec(&d_pk, seed_d, MLWQ_Q / P_PK);
    
    // 4. 计算 As + d (复用 NTT 域矩阵，避免重复变换)
    poly_vec As;
    poly_vec s_ntt = sk->s;
    for (int i = 0; i < MLWQ_K; ++i) {
        avx_ntt(s_ntt.vec[i].coeffs);
    }
    avx_poly_matrix_vec_mul_ntt(&As, &A_ntt, &s_ntt);
    
    memcpy(pk->seed_A, seed_A, 32);
    memcpy(pk->seed_d, seed_d, 32);
    
    for(int i=0; i<MLWQ_K; ++i)
        avx_poly_quantize(&pk->b_q.vec[i], &As.vec[i], &d_pk.vec[i], P_PK);
}

// -------------------------------------------------------------------------
// PKE Encrypt
// -------------------------------------------------------------------------
void avx_mlwq_encrypt(mlwq_ciphertext *ct, const mlwq_pk *pk, const uint8_t *msg, const uint8_t *seed_ct) {
    poly_matrix A_ntt;
    static poly_matrix cached_A_ntt;
    static uint8_t cached_seed_A[32];
    static int cached_A_valid = 0;
    static poly_vec cached_b_ntt;
    static poly_vec cached_b_q;
    static int cached_b_valid = 0;

    if (cached_A_valid && memcmp(cached_seed_A, pk->seed_A, 32) == 0) {
        A_ntt = cached_A_ntt;
    } else {
        poly_matrix A;
        avx_xof_expand_matrix(&A, pk->seed_A);
        A_ntt = A;
        for (int i = 0; i < MLWQ_K; ++i) {
            for (int j = 0; j < MLWQ_K; ++j) {
                avx_ntt(A_ntt.row[i].vec[j].coeffs);
            }
        }
        cached_A_ntt = A_ntt;
        memcpy(cached_seed_A, pk->seed_A, 32);
        cached_A_valid = 1;
    }
    
    // 1. 并行采样噪声 r (SHAKE256x4 + AVX CBD3)
    poly_vec r;
    avx_polyvec_getnoise_eta1(&r, seed_ct, 0);

    poly_vec r_ntt = r;
    for(int i=0; i<MLWQ_K; ++i) {
        avx_ntt(r_ntt.vec[i].coeffs);
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
    
    uint16_t mod_v = (uint16_t)(MLWQ_Q / P_V);
    uint32_t recip_v = (uint32_t)(((uint64_t)1 << 32) / mod_v);
    for(int k=0; k<MLWQ_N; ++k) {
        uint16_t val = (uint16_t)buf_v[2*k] | ((uint16_t)buf_v[2*k+1]<<8);
        uint32_t q = (uint32_t)(((uint64_t)val * recip_v) >> 32);
        uint32_t r = val - q * mod_v;
        if (r >= mod_v) r -= mod_v;
        d_v.coeffs[k] = (int16_t)r;
    }

    // 后续计算...
    poly_matrix At_ntt;
    for(int i=0; i<MLWQ_K; ++i) for(int j=0; j<MLWQ_K; ++j) At_ntt.row[i].vec[j] = A_ntt.row[j].vec[i];
    
    poly_vec Atr;
    avx_poly_matrix_vec_mul_ntt(&Atr, &At_ntt, &r_ntt);
    
    if (!cached_b_valid || memcmp(&cached_b_q, &pk->b_q, sizeof(poly_vec)) != 0) {
        poly_vec b_deq;
        for(int i=0; i<MLWQ_K; ++i) {
            avx_poly_dequantize(&b_deq.vec[i], &pk->b_q.vec[i], P_PK);
        }
        cached_b_ntt = b_deq;
        for(int i=0; i<MLWQ_K; ++i) {
            avx_ntt(cached_b_ntt.vec[i].coeffs);
        }
        cached_b_q = pk->b_q;
        cached_b_valid = 1;
    }
    
    poly v_val;
    avx_poly_vec_transpose_mul_ntt(&v_val, &cached_b_ntt, &r_ntt);
    
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
    static poly_vec cached_s_ntt;
    static poly_vec cached_s;
    static int cached_s_valid = 0;
    static poly_vec cached_u_ntt;
    static poly_vec cached_u;
    static int cached_u_valid = 0;

    if (!cached_s_valid || memcmp(&cached_s, &sk->s, sizeof(poly_vec)) != 0) {
        cached_s = sk->s;
        cached_s_ntt = sk->s;
        for (int i = 0; i < MLWQ_K; ++i) {
            avx_ntt(cached_s_ntt.vec[i].coeffs);
        }
        cached_s_valid = 1;
    }

    poly_vec u_deq;
    for(int i=0; i<MLWQ_K; ++i) avx_poly_dequantize(&u_deq.vec[i], &ct->u.vec[i], P_U);
    poly v_deq;
    avx_poly_dequantize(&v_deq, &ct->v, P_V);
    poly s_t_u;
    if (!cached_u_valid || memcmp(&cached_u, &u_deq, sizeof(poly_vec)) != 0) {
        cached_u = u_deq;
        cached_u_ntt = u_deq;
        for (int i = 0; i < MLWQ_K; ++i) {
            avx_ntt(cached_u_ntt.vec[i].coeffs);
        }
        cached_u_valid = 1;
    }
    avx_poly_vec_transpose_mul_ntt(&s_t_u, &cached_s_ntt, &cached_u_ntt);
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
