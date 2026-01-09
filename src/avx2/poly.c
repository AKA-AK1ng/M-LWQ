#include "poly.h"
#include "ntt.h"
#include <immintrin.h>
#include <string.h>

#define MLWQ_Q 3329

// [AVX2] 向量模加
static inline __m256i avx_add_mod(__m256i a, __m256i b) {
    __m256i vq = _mm256_set1_epi16(MLWQ_Q);
    __m256i sum = _mm256_add_epi16(a, b);
    __m256i mask = _mm256_cmpgt_epi16(sum, _mm256_set1_epi16(MLWQ_Q - 1));
    return _mm256_sub_epi16(sum, _mm256_and_si256(mask, vq));
}

// [AVX2] 向量模减
static inline __m256i avx_sub_mod(__m256i a, __m256i b) {
    __m256i vq = _mm256_set1_epi16(MLWQ_Q);
    __m256i diff = _mm256_sub_epi16(a, b);
    __m256i mask = _mm256_srai_epi16(diff, 15);
    return _mm256_add_epi16(diff, _mm256_and_si256(mask, vq));
}

static inline void avx_poly_add_inplace(poly *res, const poly *b) {
    for(int i=0; i<MLWQ_N/16; ++i) {
        __m256i va = _mm256_load_si256((__m256i*)&res->coeffs[16*i]);
        __m256i vb = _mm256_load_si256((__m256i*)&b->coeffs[16*i]);
        _mm256_store_si256((__m256i*)&res->coeffs[16*i], avx_add_mod(va, vb));
    }
}

static inline void avx_basemul_acc(int16_t *acc, const int16_t *a, const int16_t *b) {
    __m256i prod[MLWQ_N / 16] __attribute__((aligned(32)));
    avx_basemul((int16_t *)prod, a, b);
    for(int i=0; i<MLWQ_N/16; ++i) {
        __m256i va = _mm256_load_si256((__m256i*)&acc[16*i]);
        __m256i vb = _mm256_load_si256(&prod[i]);
        _mm256_store_si256((__m256i*)&acc[16*i], avx_add_mod(va, vb));
    }
}

void avx_poly_add(poly *res, const poly *a, const poly *b) {
    for(int i=0; i<MLWQ_N/16; ++i) {
        __m256i va = _mm256_load_si256((__m256i*)&a->coeffs[16*i]);
        __m256i vb = _mm256_load_si256((__m256i*)&b->coeffs[16*i]);
        _mm256_store_si256((__m256i*)&res->coeffs[16*i], avx_add_mod(va, vb));
    }
}

void avx_poly_sub(poly *res, const poly *a, const poly *b) {
    for(int i=0; i<MLWQ_N/16; ++i) {
        __m256i va = _mm256_load_si256((__m256i*)&a->coeffs[16*i]);
        __m256i vb = _mm256_load_si256((__m256i*)&b->coeffs[16*i]);
        _mm256_store_si256((__m256i*)&res->coeffs[16*i], avx_sub_mod(va, vb));
    }
}

// [FIX] AVX2 极速量化: 消除除法，使用 vpmulhuw
void avx_poly_quantize(poly *res, const poly *v, const poly *d, int32_t P) {
    // 预计算定点乘法因子 magic = floor(P * 2^16 / Q)
    // 这样 (x * magic) >> 16 等价于 x * P / Q
    int32_t magic_val = (P * 65536) / MLWQ_Q;
    __m256i magic = _mm256_set1_epi16((int16_t)magic_val);
    __m256i mask_vec = _mm256_set1_epi16((int16_t)(P - 1));

    for(int i=0; i<MLWQ_N/16; ++i) {
        __m256i vv = _mm256_load_si256((__m256i*)&v->coeffs[16*i]);
        __m256i vd = _mm256_load_si256((__m256i*)&d->coeffs[16*i]);
        
        // 1. sum = v + d (直接加，暂不取模，因为后面要乘 P/Q)
        // 注意：v 和 d 都是 [0, Q) 范围，相加最大 ~6658，远小于 int16 上限
        __m256i sum = _mm256_add_epi16(vv, vd);

        // 2. High Multiply: (sum * magic) >> 16
        // _mm256_mulhi_epu16 是无符号高位乘法，完美符合 (a*b)>>16
        __m256i floor_val = _mm256_mulhi_epu16(sum, magic);

        // 3. Mask
        __m256i q_res = _mm256_and_si256(floor_val, mask_vec);

        _mm256_store_si256((__m256i*)&res->coeffs[16*i], q_res);
    }
}

// [FIX] AVX2 极速反量化: v = (b * Q + Q/2) / P
// 近似为: v = (b * floor(Q*2^16/P)) >> 16
void avx_poly_dequantize(poly *res, const poly *b, int32_t P) {
    // v = (b * Q + Q/2) / P, optimized for power-of-two P (32/512/1024)
    int shift = -1;
    if (P == 32) {
        shift = 5;
    } else if (P == 512) {
        shift = 9;
    } else if (P == 1024) {
        shift = 10;
    }

    if (shift < 0) {
        for (int i = 0; i < MLWQ_N; ++i) {
            int32_t v = (int32_t)b->coeffs[i] * MLWQ_Q + (MLWQ_Q / 2);
            res->coeffs[i] = v / P;
        }
        return;
    }

    __m256i q = _mm256_set1_epi32(MLWQ_Q);
    __m256i half_q = _mm256_set1_epi32(MLWQ_Q / 2);

    for (int i = 0; i < MLWQ_N; i += 16) {
        __m256i v = _mm256_load_si256((__m256i *)&b->coeffs[i]);
        __m256i v_lo = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(v));
        __m256i v_hi = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(v, 1));

        v_lo = _mm256_mullo_epi32(v_lo, q);
        v_hi = _mm256_mullo_epi32(v_hi, q);
        v_lo = _mm256_add_epi32(v_lo, half_q);
        v_hi = _mm256_add_epi32(v_hi, half_q);
        v_lo = _mm256_srli_epi32(v_lo, shift);
        v_hi = _mm256_srli_epi32(v_hi, shift);

        __m256i packed = _mm256_packus_epi32(v_lo, v_hi);
        packed = _mm256_permute4x64_epi64(packed, 0xD8);
        _mm256_store_si256((__m256i *)&res->coeffs[i], packed);
    }
}

void avx_poly_matrix_vec_mul(poly_vec *res, const poly_matrix *A, const poly_vec *s) {
    poly_vec s_ntt = *s;
    for(int j=0; j<MLWQ_K; ++j) {
        avx_ntt(s_ntt.vec[j].coeffs);
    }

    for(int i=0; i<MLWQ_K; ++i) {
        poly acc;
        poly a_ntt = A->row[i].vec[0];
        avx_ntt(a_ntt.coeffs);
        avx_basemul(acc.coeffs, a_ntt.coeffs, s_ntt.vec[0].coeffs);
        for(int j=1; j<MLWQ_K; ++j) {
            poly a_ntt = A->row[i].vec[j];
            poly prod;
            avx_ntt(a_ntt.coeffs);
            avx_basemul(prod.coeffs, a_ntt.coeffs, s_ntt.vec[j].coeffs);
            avx_poly_add_inplace(&acc, &prod);
        }
        avx_invntt(acc.coeffs);
        avx_reduce(acc.coeffs);
        res->vec[i] = acc;
    }
}

void avx_poly_matrix_vec_mul_ntt(poly_vec *res, const poly_matrix *A_ntt, const poly_vec *s_ntt) {
    for(int i=0; i<MLWQ_K; ++i) {
        poly acc;
        poly prod;
        avx_basemul(acc.coeffs, A_ntt->row[i].vec[0].coeffs, s_ntt->vec[0].coeffs);
        for(int j=1; j<MLWQ_K; ++j) {
            avx_basemul(prod.coeffs, A_ntt->row[i].vec[j].coeffs, s_ntt->vec[j].coeffs);
            avx_poly_add_inplace(&acc, &prod);
        }
        avx_invntt(acc.coeffs);
        avx_reduce(acc.coeffs);
        res->vec[i] = acc;
    }
}

void avx_polyvec_basemul_acc(poly *res, const poly_vec *a, const poly_vec *b) {
#if MLWQ_K == 2
    avx_basemul(res->coeffs, a->vec[0].coeffs, b->vec[0].coeffs);
    avx_basemul_acc(res->coeffs, a->vec[1].coeffs, b->vec[1].coeffs);
#elif MLWQ_K == 3
    avx_basemul(res->coeffs, a->vec[0].coeffs, b->vec[0].coeffs);
    avx_basemul_acc(res->coeffs, a->vec[1].coeffs, b->vec[1].coeffs);
    avx_basemul_acc(res->coeffs, a->vec[2].coeffs, b->vec[2].coeffs);
#elif MLWQ_K == 4
    avx_basemul(res->coeffs, a->vec[0].coeffs, b->vec[0].coeffs);
    avx_basemul_acc(res->coeffs, a->vec[1].coeffs, b->vec[1].coeffs);
    avx_basemul_acc(res->coeffs, a->vec[2].coeffs, b->vec[2].coeffs);
    avx_basemul_acc(res->coeffs, a->vec[3].coeffs, b->vec[3].coeffs);
#else
    avx_basemul(res->coeffs, a->vec[0].coeffs, b->vec[0].coeffs);
    for(int i=1; i<MLWQ_K; ++i) {
        avx_basemul_acc(res->coeffs, a->vec[i].coeffs, b->vec[i].coeffs);
    }
#endif
}

void avx_poly_vec_transpose_mul(poly *res, const poly_vec *a_t, const poly_vec *b) {
    poly_vec a_ntt = *a_t;
    poly_vec b_ntt = *b;
    for(int i=0; i<MLWQ_K; ++i) {
        avx_ntt(a_ntt.vec[i].coeffs);
        avx_ntt(b_ntt.vec[i].coeffs);
    }

    poly acc;
    poly prod;
    avx_basemul(acc.coeffs, a_ntt.vec[0].coeffs, b_ntt.vec[0].coeffs);
    for(int i=1; i<MLWQ_K; ++i) {
        avx_basemul(prod.coeffs, a_ntt.vec[i].coeffs, b_ntt.vec[i].coeffs);
        avx_poly_add_inplace(&acc, &prod);
    }
    avx_invntt(acc.coeffs);
    avx_reduce(acc.coeffs);
    *res = acc;
}

void avx_poly_vec_transpose_mul_ntt(poly *res, const poly_vec *a_ntt, const poly_vec *b_ntt) {
    poly acc;
    poly prod;
    avx_basemul(acc.coeffs, a_ntt->vec[0].coeffs, b_ntt->vec[0].coeffs);
    for(int i=1; i<MLWQ_K; ++i) {
        avx_basemul(prod.coeffs, a_ntt->vec[i].coeffs, b_ntt->vec[i].coeffs);
        avx_poly_add_inplace(&acc, &prod);
    }
    avx_invntt(acc.coeffs);
    avx_reduce(acc.coeffs);
    *res = acc;
}

void avx_poly_msg_encode(poly *res, const uint8_t *msg) {
    static __m128i table128[256];
    static int table_ready = 0;
    if (!table_ready) {
        int16_t scale = MLWQ_Q / 2;
        for (int v = 0; v < 256; v++) {
            int16_t tmp[8];
            for (int b = 0; b < 8; b++) {
                tmp[b] = (v & (1 << b)) ? scale : 0;
            }
            table128[v] = _mm_loadu_si128((const __m128i *)tmp);
        }
        table_ready = 1;
    }

    for (int i = 0; i < MLWQ_N / 16; i++) {
        __m128i lo = table128[msg[2 * i]];
        __m128i hi = table128[msg[2 * i + 1]];
        __m256i packed = _mm256_castsi128_si256(lo);
        packed = _mm256_inserti128_si256(packed, hi, 1);
        _mm256_store_si256((__m256i *)&res->coeffs[16 * i], packed);
    }
}

void avx_poly_msg_decode(uint8_t *msg, const poly *p) {
    static uint8_t compress_lut[256];
    static int table_ready = 0;
    if (!table_ready) {
        for (int v = 0; v < 256; v++) {
            uint8_t out = 0;
            out |= (uint8_t)((v >> 0) & 0x1);
            out |= (uint8_t)(((v >> 2) & 0x1) << 1);
            out |= (uint8_t)(((v >> 4) & 0x1) << 2);
            out |= (uint8_t)(((v >> 6) & 0x1) << 3);
            compress_lut[v] = out;
        }
        table_ready = 1;
    }

    memset(msg, 0, 32);
    __m256i lower = _mm256_set1_epi16((int16_t)(MLWQ_Q / 4));
    __m256i upper = _mm256_set1_epi16((int16_t)(3 * MLWQ_Q / 4));

    for (int i = 0; i < MLWQ_N / 16; i++) {
        __m256i v = _mm256_load_si256((const __m256i *)&p->coeffs[16 * i]);
        __m256i gt = _mm256_cmpgt_epi16(v, lower);
        __m256i lt = _mm256_cmpgt_epi16(upper, v);
        __m256i in_range = _mm256_and_si256(gt, lt);

        uint32_t mask = (uint32_t)_mm256_movemask_epi8(in_range);
        uint8_t b0 = (uint8_t)(mask & 0xFFu);
        uint8_t b1 = (uint8_t)((mask >> 8) & 0xFFu);
        uint8_t b2 = (uint8_t)((mask >> 16) & 0xFFu);
        uint8_t b3 = (uint8_t)((mask >> 24) & 0xFFu);

        msg[2 * i] = (uint8_t)(compress_lut[b0] | (compress_lut[b1] << 4));
        msg[2 * i + 1] = (uint8_t)(compress_lut[b2] | (compress_lut[b3] << 4));
    }
}

int avx_check_poly_eq(const poly *a, const poly *b) {
    return memcmp(a, b, sizeof(poly)) == 0;
}
