#include "ntt_avx512.h"
#include "../avx2/ntt.h"

void avx512_ntt(int16_t *r) {
    avx_ntt(r);
}

void avx512_invntt(int16_t *r) {
    avx_invntt(r);
}

void avx512_basemul(int16_t *r, const int16_t *a, const int16_t *b) {
    avx_basemul(r, a, b);
}

void avx512_reduce(int16_t *r) {
    avx_reduce(r);
}

void avx512_poly_mul_ntt(poly *res, const poly *a, const poly *b) {
    avx_poly_mul_ntt(res, a, b);
}
