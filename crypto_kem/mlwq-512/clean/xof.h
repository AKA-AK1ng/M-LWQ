#ifndef XOF_H
#define XOF_H

#include "structs.h"

void xof_expand_matrix(poly_matrix *A, const uint8_t *seed);
void xof_expand_poly_vec(poly_vec *v, const uint8_t *seed, int32_t modulus);

#endif
