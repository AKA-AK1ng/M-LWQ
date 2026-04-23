#include <stdint.h>
#include <string.h>
#include "xof.h"
#include "params.h"
#include "fips202.h"

// -------------------------------------------------------------------------
// Helper: rejection sampling for uniform coefficients in [0, Q)
// -------------------------------------------------------------------------
static unsigned int rej_uniform(int16_t *r,
                                unsigned int len,
                                const uint8_t *buf,
                                unsigned int buflen)
{
  unsigned int ctr, pos;
  uint16_t val0, val1;

  ctr = pos = 0;
  while(ctr < len && pos + 3 <= buflen) {
    val0 = ((buf[pos+0] >> 0) | ((uint16_t)buf[pos+1] << 8)) & 0xFFF;
    val1 = ((buf[pos+1] >> 4) | ((uint16_t)buf[pos+2] << 4)) & 0xFFF;
    pos += 3;

    if(val0 < MLWQ_Q)
      r[ctr++] = val0;
    if(ctr < len && val1 < MLWQ_Q)
      r[ctr++] = val1;
  }

  return ctr;
}

// -------------------------------------------------------------------------
// Matrix generation (A)
// -------------------------------------------------------------------------
#define GEN_MATRIX_NBLOCKS 3

void xof_expand_matrix(poly_matrix *A, const uint8_t *seed) {
  unsigned int i, j;
  unsigned int buflen;
  unsigned int ctr;
  uint8_t buf[GEN_MATRIX_NBLOCKS * SHAKE128_RATE];
  uint8_t extseed[34];
  keccak_state state;

  for(i=0; i<MLWQ_K; i++) {
    for(j=0; j<MLWQ_K; j++) {
      memcpy(extseed, seed, 32);
      extseed[32] = j;
      extseed[33] = i;

      shake128_absorb_once(&state, extseed, 34);
      shake128_squeezeblocks(buf, GEN_MATRIX_NBLOCKS, &state);
      buflen = GEN_MATRIX_NBLOCKS * SHAKE128_RATE;

      ctr = rej_uniform(A->row[i].vec[j].coeffs, MLWQ_N, buf, buflen);

      while(ctr < MLWQ_N) {
        shake128_squeezeblocks(buf, 1, &state);
        buflen = SHAKE128_RATE;
        ctr += rej_uniform(A->row[i].vec[j].coeffs + ctr, MLWQ_N - ctr, buf, buflen);
      }
    }
  }
}

// -------------------------------------------------------------------------
// Vector generation (Dither)
// -------------------------------------------------------------------------
static unsigned int rej_uniform_mod(int16_t *r,
                                    unsigned int len,
                                    const uint8_t *buf,
                                    unsigned int buflen,
                                    int32_t modulus)
{
  unsigned int ctr = 0;
  unsigned int pos = 0;

  while(ctr < len && pos + 2 <= buflen) {
      uint16_t val = (uint16_t)(buf[pos]) | ((uint16_t)(buf[pos+1]) << 8);
      pos += 2;
      r[ctr++] = val % modulus;
  }
  return ctr;
}

void xof_expand_poly_vec(poly_vec *v, const uint8_t *seed, int32_t modulus) {
  unsigned int i;
  unsigned int ctr;
  unsigned int buflen;
  uint8_t buf[2 * SHAKE128_RATE];
  uint8_t extseed[33];
  keccak_state state;

  for(i=0; i<MLWQ_K; i++) {
    memcpy(extseed, seed, 32);
    extseed[32] = i;

    shake128_absorb_once(&state, extseed, 33);
    shake128_squeezeblocks(buf, 2, &state);
    buflen = 2 * SHAKE128_RATE;

    ctr = rej_uniform_mod(v->vec[i].coeffs, MLWQ_N, buf, buflen, modulus);

    while(ctr < MLWQ_N) {
        shake128_squeezeblocks(buf, 1, &state);
        buflen = SHAKE128_RATE;
        ctr += rej_uniform_mod(v->vec[i].coeffs + ctr, MLWQ_N - ctr, buf, buflen, modulus);
    }
  }
}
