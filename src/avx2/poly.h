#ifndef AVX_POLY_H
#define AVX_POLY_H

#include "../common/structs.h"

void avx_poly_add(poly *res, const poly *a, const poly *b);
void avx_poly_sub(poly *res, const poly *a, const poly *b);
void avx_poly_quantize(poly *res, const poly *v, const poly *d, int32_t P);
void avx_poly_dequantize(poly *res, const poly *b, int32_t P);

void avx_poly_vec_transpose_mul(poly *res, const poly_vec *a_t, const poly_vec *b);
void avx_poly_matrix_vec_mul(poly_vec *res, const poly_matrix *A, const poly_vec *s);

void avx_poly_msg_encode(poly *res, const uint8_t *msg);
void avx_poly_msg_decode(uint8_t *msg, const poly *p);

// 检查函数 (一般不做 AVX 加速，直接调用 memcmp 或标量)
int avx_check_poly_eq(const poly *a, const poly *b);

#endif