#include "random.h"
#include <stdlib.h>
#include <time.h>

void random_init() {
    srand(time(NULL));
}

void random_bytes(uint8_t *out, size_t len) {
    for(size_t i=0; i<len; ++i) out[i] = rand() & 0xFF;
}

void random_poly_uniform(poly *p) {
    for(int i=0; i<MLWQ_N; ++i) {
        p->coeffs[i] = (int16_t)(rand() % MLWQ_Q);
    }
}

// [FIX] 使用 CBD2 (Centered Binomial Distribution)
// 之前是 Uniform(5)，噪声太大
void random_poly_eta(poly *p) {
    for(int i=0; i<MLWQ_N; ++i) {
        // 模拟 CBD2: [-2, 2]
        // 概率: 0(37.5%), 1(25%), 2(6.25%)
        int b0 = rand() & 1;
        int b1 = rand() & 1;
        int b2 = rand() & 1;
        int b3 = rand() & 1;
        p->coeffs[i] = (b0 + b1) - (b2 + b3);
    }
}

void random_poly_vec_eta(poly_vec *pv) {
    for(int i=0; i<MLWQ_K; ++i) {
        random_poly_eta(&pv->vec[i]);
    }
}