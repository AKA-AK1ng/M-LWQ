#ifndef POLYVEC_H
#define POLYVEC_H

#include <stdint.h>
#include "poly.h"
#include "../common/params.h"

// -------------------------------------------------------------------------
// 结构体定义
// -------------------------------------------------------------------------

// 1. 定义核心向量结构
typedef struct {
    poly vec[MLWQ_K];
} polyvec;

// [修复关键点] 为了兼容旧代码，定义别名 poly_vec -> polyvec
typedef polyvec poly_vec;

// 2. 定义矩阵结构
typedef struct {
    polyvec row[MLWQ_K];
} poly_matrix;

// -------------------------------------------------------------------------
// 函数声明
// -------------------------------------------------------------------------

// 序列化
void ref_polyvec_tobytes(uint8_t r[MLWQ_POLYVECBYTES], const polyvec *a);
void ref_polyvec_frombytes(polyvec *r, const uint8_t a[MLWQ_POLYVECBYTES]);

// 算术
void ref_polyvec_ntt(polyvec *r);
void ref_polyvec_invntt(polyvec *r);
void ref_polyvec_add(polyvec *r, const polyvec *a, const polyvec *b);

// 矩阵运算 (这些函数的实现在 poly.c 中，但在这里声明比较合理，或者在 poly.h)
// 注意：为了解决 poly.c 的报错，我们需要确保这些类型可见
void ref_poly_matrix_vec_mul(polyvec *res, const poly_matrix *A, const polyvec *s);
void ref_poly_vec_transpose_mul(poly *res, const polyvec *a_t, const polyvec *b);

#endif