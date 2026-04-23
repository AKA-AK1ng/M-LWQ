#ifndef POLY_H
#define POLY_H

#include "structs.h"

void poly_add(poly *res, const poly *a, const poly *b);
void poly_sub(poly *res, const poly *a, const poly *b);

void poly_quantize(poly *res, const poly *v, const poly *d, int32_t P);
void poly_dequantize(poly *res, const poly *b, int32_t P);

void poly_vec_transpose_mul(poly *res, const poly_vec *a_t, const poly_vec *b);
void poly_matrix_vec_mul(poly_vec *res, const poly_matrix *A, const poly_vec *s);

void poly_msg_encode(poly *res, const uint8_t *msg);
void poly_msg_decode(uint8_t *msg, const poly *p);

void poly_mul_ntt(poly *res, const poly *a, const poly *b);

void poly_tobytes(uint8_t *r, const poly *a);
void poly_frombytes(poly *r, const uint8_t *a);

void poly_compress_u(uint8_t *r, const poly *a);
void poly_decompress_u(poly *r, const uint8_t *a);

void poly_compress_v(uint8_t *r, const poly *a);
void poly_decompress_v(poly *r, const uint8_t *a);

void poly_getnoise_eta1(poly *r, const uint8_t *seed, uint8_t nonce);

#endif
