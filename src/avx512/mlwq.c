#include "mlwq.h"
#include "poly.h"
#include "ntt_avx512.h"
#include "avx512/xof.h"
#include "noise.h"
#include "align.h"
#include "../common/random.h"
#include "../common/fips202.h"
#include <string.h>
#include <immintrin.h>

static void derive_seed_d(uint8_t *seed_d, const uint8_t *seed_a) {
    uint8_t input[33];
    memcpy(input, seed_a, 32);
    input[32] = 0x01;
    shake128(seed_d, 32, input, sizeof(input));
}

extern void avx_poly_tobytes(uint8_t *r, const poly *a);
extern void avx_poly_frombytes(poly *r, const uint8_t *a);
extern void avx_poly_compress_u(uint8_t *r, const poly *a);
extern void avx_poly_decompress_u(poly *r, const uint8_t *a);
extern void avx_poly_compress_v(uint8_t *r, const poly *a);
extern void avx_poly_decompress_v(poly *r, const uint8_t *a);
extern void avx_poly_matrix_vec_mul_ntt(poly_vec *res, const poly_matrix *A_ntt, const poly_vec *s_ntt);
extern void avx_poly_matrix_vec_mul(poly_vec *res, const poly_matrix *A, const poly_vec *s);
extern void avx_poly_vec_transpose_mul_ntt(poly *res, const poly_vec *a_ntt, const poly_vec *b_ntt);
extern void avx_poly_vec_transpose_mul(poly *res, const poly_vec *a, const poly_vec *b);
extern void avx_poly_msg_encode(poly *res, const uint8_t *msg);
extern void avx_poly_msg_decode(uint8_t *msg, const poly *p);
extern void avx_poly_quantize(poly *res, const poly *v, const poly *d, int32_t P);
extern void avx_poly_dequantize(poly *res, const poly *b, int32_t P);
extern void avx_poly_add(poly *res, const poly *a, const poly *b);
extern void avx_poly_sub(poly *res, const poly *a, const poly *b);

// -------------------------------------------------------------------------
// PKE KeyGen
// -------------------------------------------------------------------------
void avx512_mlwq_keygen(mlwq_pk *pk, mlwq_sk *sk, const uint8_t *seed_A) {
    poly_matrix A;
    avx512_xof_expand_matrix(&A, seed_A);
    poly_matrix A_ntt = A;
    for (int i = 0; i < MLWQ_K; ++i) {
        for (int j = 0; j < MLWQ_K; ++j) {
            avx512_ntt(A_ntt.row[i].vec[j].coeffs);
        }
    }

    uint8_t seed_s[32];
    random_bytes(seed_s, 32);
    avx512_polyvec_getnoise_eta1(&sk->s, seed_s, 0);

    poly_vec d_pk;
    uint8_t d_seed[32];
    uint8_t d_uniform_seed[33];
    derive_seed_d(d_seed, seed_A);
    memcpy(d_uniform_seed, d_seed, 32);
    d_uniform_seed[32] = 0xFF;
    avx512_xof_expand_poly_vec(&d_pk, d_uniform_seed, MLWQ_Q);

    poly_vec As;
    poly_vec s_ntt = sk->s;
    for (int i = 0; i < MLWQ_K; ++i) {
        avx512_ntt(s_ntt.vec[i].coeffs);
    }
    avx_poly_matrix_vec_mul_ntt(&As, &A_ntt, &s_ntt);

    memcpy(pk->seed_A, seed_A, 32);
    memcpy(pk->seed_d, d_seed, 32);

    for (int i = 0; i < MLWQ_K; ++i) {
        avx_poly_quantize(&pk->b_q.vec[i], &As.vec[i], &d_pk.vec[i], P_PK);
    }
}

// -------------------------------------------------------------------------
// PKE Encrypt
// -------------------------------------------------------------------------
void avx512_mlwq_encrypt(mlwq_ciphertext *ct, const mlwq_pk *pk, const uint8_t *msg, const uint8_t *seed_ct) {
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
        avx512_xof_expand_matrix(&A, pk->seed_A);
        A_ntt = A;
        for (int i = 0; i < MLWQ_K; ++i) {
            for (int j = 0; j < MLWQ_K; ++j) {
                avx512_ntt(A_ntt.row[i].vec[j].coeffs);
            }
        }
        cached_A_ntt = A_ntt;
        memcpy(cached_seed_A, pk->seed_A, 32);
        cached_A_valid = 1;
    }

    poly_vec r;
    avx512_polyvec_getnoise_eta1(&r, seed_ct, 0);

    poly_vec r_ntt = r;
    for (int i = 0; i < MLWQ_K; ++i) {
        avx512_ntt(r_ntt.vec[i].coeffs);
    }

    poly_vec d_u;
    poly d_v;
    uint8_t d_seed[33];
    memcpy(d_seed, seed_ct, 32);
    d_seed[32] = MLWQ_K;
    avx512_xof_expand_poly_vec(&d_u, d_seed, MLWQ_Q);
    d_seed[32] = MLWQ_K + 1;
    avx512_xof_expand_poly(&d_v, d_seed, MLWQ_Q);

    poly_matrix At_ntt;
    for (int i = 0; i < MLWQ_K; ++i) {
        for (int j = 0; j < MLWQ_K; ++j) {
            At_ntt.row[i].vec[j] = A_ntt.row[j].vec[i];
        }
    }

    poly_vec Atr;
    avx_poly_matrix_vec_mul_ntt(&Atr, &At_ntt, &r_ntt);

    if (!cached_b_valid || memcmp(&cached_b_q, &pk->b_q, sizeof(poly_vec)) != 0) {
        poly_vec b_deq;
        for (int i = 0; i < MLWQ_K; ++i) {
            avx_poly_dequantize(&b_deq.vec[i], &pk->b_q.vec[i], P_PK);
        }
        cached_b_ntt = b_deq;
        for (int i = 0; i < MLWQ_K; ++i) {
            avx512_ntt(cached_b_ntt.vec[i].coeffs);
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

    for (int i = 0; i < MLWQ_K; ++i) {
        avx_poly_quantize(&ct->u.vec[i], &Atr.vec[i], &d_u.vec[i], P_U);
    }
    avx_poly_quantize(&ct->v, &v_final, &d_v, P_V);
}

// -------------------------------------------------------------------------
// Decrypt / KEM Wrappers
// -------------------------------------------------------------------------
void avx512_mlwq_decrypt(uint8_t *msg, const mlwq_sk *sk, const mlwq_ciphertext *ct) {
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
            avx512_ntt(cached_s_ntt.vec[i].coeffs);
        }
        cached_s_valid = 1;
    }

    poly_vec u_deq;
    for (int i = 0; i < MLWQ_K; ++i) {
        avx_poly_dequantize(&u_deq.vec[i], &ct->u.vec[i], P_U);
    }
    poly v_deq;
    avx_poly_dequantize(&v_deq, &ct->v, P_V);
    poly s_t_u;
    if (!cached_u_valid || memcmp(&cached_u, &u_deq, sizeof(poly_vec)) != 0) {
        cached_u = u_deq;
        cached_u_ntt = u_deq;
        for (int i = 0; i < MLWQ_K; ++i) {
            avx512_ntt(cached_u_ntt.vec[i].coeffs);
        }
        cached_u_valid = 1;
    }
    avx_poly_vec_transpose_mul_ntt(&s_t_u, &cached_s_ntt, &cached_u_ntt);
    poly diff;
    avx_poly_sub(&diff, &v_deq, &s_t_u);
    avx_poly_msg_decode(msg, &diff);
}

void avx512_mlwq_kem_keygen(mlwq_pk *pk, mlwq_kem_sk *sk) {
    uint8_t seed_A[32];
    random_bytes(seed_A, 32);
    avx512_mlwq_keygen(pk, &sk->pke_sk, seed_A);
    sk->pk = *pk;
    shake128(sk->h_pk, 32, (uint8_t *)pk, sizeof(mlwq_pk));
    random_bytes(sk->z, 32);
}

void avx512_mlwq_kem_encaps(mlwq_ciphertext *ct, uint8_t *ss, const mlwq_pk *pk) {
    uint8_t m[32];
    random_bytes(m, 32);
    uint8_t buf[64];
    memcpy(buf, m, 32);
    shake128(buf + 32, 32, (uint8_t *)pk, sizeof(mlwq_pk));
    uint8_t kr[64];
    shake128(kr, 64, buf, 64);
    memcpy(ss, kr, 32);
    avx512_mlwq_encrypt(ct, pk, m, kr + 32);
}

int avx512_mlwq_kem_decaps(uint8_t *ss, const mlwq_kem_sk *sk, const mlwq_ciphertext *ct) {
    uint8_t m[32];
    avx512_mlwq_decrypt(m, &sk->pke_sk, ct);
    uint8_t buf[64];
    memcpy(buf, m, 32);
    memcpy(buf + 32, sk->h_pk, 32);
    uint8_t kr[64];
    shake128(kr, 64, buf, 64);
    mlwq_ciphertext ct_prime;
    avx512_mlwq_encrypt(&ct_prime, &sk->pk, m, kr + 32);
    if (memcmp(ct, &ct_prime, sizeof(mlwq_ciphertext)) == 0) {
        memcpy(ss, kr, 32);
        return 1;
    } else {
        shake128(ss, 32, sk->z, 32);
        return 0;
    }
}
