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
    // 这里为了精度，通常还是用 float 或者 wider int
    // 但为了 AVX2 速度，且 P 通常是 2 的幂，可以直接用移位
    // M-LWQ 中 P 是 1024 或 512 等
    // v = (b * Q) / P = b * (Q/P)
    // 如果 Q/P 是整数（不一定），或者用定点
    // 简单起见，这里保持标量循环，因为它不是性能热点，且精度要求高
    for(int i=0; i<MLWQ_N; ++i) {
        int32_t v = (int32_t)b->coeffs[i] * MLWQ_Q + (MLWQ_Q/2);
        res->coeffs[i] = v / P;
    }
}

void avx_poly_matrix_vec_mul(poly_vec *res, const poly_matrix *A, const poly_vec *s) {
    poly_vec s_ntt = *s;
    for(int j=0; j<MLWQ_K; ++j) {
        avx_ntt(s_ntt.vec[j].coeffs);
    }

    for(int i=0; i<MLWQ_K; ++i) {
        poly acc;
        memset(acc.coeffs, 0, sizeof(int16_t)*MLWQ_N);
        for(int j=0; j<MLWQ_K; ++j) {
            poly a_ntt = A->row[i].vec[j];
            poly prod;
            avx_ntt(a_ntt.coeffs);
            avx_basemul(prod.coeffs, a_ntt.coeffs, s_ntt.vec[j].coeffs);
            avx_poly_add_inplace(&acc, &prod);
        }
        avx_invntt(acc.coeffs);
        for(int k=0; k<MLWQ_N; ++k) {
            int16_t val = acc.coeffs[k];
            int32_t t = (int32_t)val;
            while (t < 0) t += MLWQ_Q;
            while (t >= MLWQ_Q) t -= MLWQ_Q;
            res->vec[i].coeffs[k] = (int16_t)t;
        }
    }
}

void avx_poly_matrix_vec_mul_ntt(poly_vec *res, const poly_matrix *A_ntt, const poly_vec *s_ntt) {
    for(int i=0; i<MLWQ_K; ++i) {
        poly acc;
        memset(acc.coeffs, 0, sizeof(int16_t)*MLWQ_N);
        for(int j=0; j<MLWQ_K; ++j) {
            poly prod;
            avx_basemul(prod.coeffs, A_ntt->row[i].vec[j].coeffs, s_ntt->vec[j].coeffs);
            avx_poly_add_inplace(&acc, &prod);
        }
        avx_invntt(acc.coeffs);
        for(int k=0; k<MLWQ_N; ++k) {
            int16_t val = acc.coeffs[k];
            int32_t t = (int32_t)val;
            while (t < 0) t += MLWQ_Q;
            while (t >= MLWQ_Q) t -= MLWQ_Q;
            res->vec[i].coeffs[k] = (int16_t)t;
        }
    }
}

void avx_polyvec_basemul_acc(poly *res, const poly_vec *a, const poly_vec *b) {
    poly acc;
    memset(acc.coeffs, 0, sizeof(int16_t)*MLWQ_N);
    for(int i=0; i<MLWQ_K; ++i) {
        poly prod;
        avx_basemul(prod.coeffs, a->vec[i].coeffs, b->vec[i].coeffs);
        avx_poly_add_inplace(&acc, &prod);
    }
    *res = acc;
}

void avx_poly_vec_transpose_mul(poly *res, const poly_vec *a_t, const poly_vec *b) {
    poly_vec a_ntt = *a_t;
    poly_vec b_ntt = *b;
    for(int i=0; i<MLWQ_K; ++i) {
        avx_ntt(a_ntt.vec[i].coeffs);
        avx_ntt(b_ntt.vec[i].coeffs);
    }

    poly acc;
    memset(acc.coeffs, 0, sizeof(int16_t)*MLWQ_N);
    for(int i=0; i<MLWQ_K; ++i) {
        poly prod;
        avx_basemul(prod.coeffs, a_ntt.vec[i].coeffs, b_ntt.vec[i].coeffs);
        avx_poly_add_inplace(&acc, &prod);
    }
    avx_invntt(acc.coeffs);
    for(int k=0; k<MLWQ_N; ++k) {
        int16_t val = acc.coeffs[k];
        int32_t t = (int32_t)val;
        while (t < 0) t += MLWQ_Q;
        while (t >= MLWQ_Q) t -= MLWQ_Q;
        res->coeffs[k] = (int16_t)t;
    }
}

void avx_poly_vec_transpose_mul_ntt(poly *res, const poly_vec *a_ntt, const poly_vec *b_ntt) {
    poly acc;
    memset(acc.coeffs, 0, sizeof(int16_t)*MLWQ_N);
    for(int i=0; i<MLWQ_K; ++i) {
        poly prod;
        avx_basemul(prod.coeffs, a_ntt->vec[i].coeffs, b_ntt->vec[i].coeffs);
        avx_poly_add_inplace(&acc, &prod);
    }
    avx_invntt(acc.coeffs);
    for(int k=0; k<MLWQ_N; ++k) {
        int16_t val = acc.coeffs[k];
        int32_t t = (int32_t)val;
        while (t < 0) t += MLWQ_Q;
        while (t >= MLWQ_Q) t -= MLWQ_Q;
        res->coeffs[k] = (int16_t)t;
    }
}

void avx_poly_msg_encode(poly *res, const uint8_t *msg) {
    static int16_t table[256][8];
    static int table_ready = 0;
    if (!table_ready) {
        int16_t scale = MLWQ_Q / 2;
        for (int v = 0; v < 256; v++) {
            for (int b = 0; b < 8; b++) {
                table[v][b] = (v & (1 << b)) ? scale : 0;
            }
        }
        table_ready = 1;
    }

    for (int i = 0; i < MLWQ_N / 8; i++) {
        const int16_t *src = table[msg[i]];
        memcpy(&res->coeffs[8 * i], src, sizeof(int16_t) * 8);
    }
}

void avx_poly_msg_decode(uint8_t *msg, const poly *p) {
    memset(msg, 0, 32);
    int32_t lower = MLWQ_Q / 4;
    int32_t upper = 3 * MLWQ_Q / 4;
    for (int i = 0; i < MLWQ_N / 8; i++) {
        const int16_t *coeffs = &p->coeffs[8 * i];
        uint8_t bits = 0;
        for (int j = 0; j < 8; j++) {
            int bit = (coeffs[j] > lower && coeffs[j] < upper) ? 1 : 0;
            bits |= (uint8_t)(bit << j);
        }
        msg[i] = bits;
    }
}

int avx_check_poly_eq(const poly *a, const poly *b) {
    return memcmp(a, b, sizeof(poly)) == 0;
}
