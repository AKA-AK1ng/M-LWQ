#ifndef REJSAMPLE_H
#define REJSAMPLE_H

#include <stdint.h>
#include "../common/fips202.h"
#include "../common/params.h"
#include "params.h"

#define REJ_UNIFORM_AVX_NBLOCKS ((12 * MLWQ_N / 8 * (1 << 12) / MLWQ_Q + SHAKE128_RATE) / SHAKE128_RATE)
#define REJ_UNIFORM_AVX_BUFLEN (REJ_UNIFORM_AVX_NBLOCKS * SHAKE128_RATE)

#define rej_uniform_avx KYBER_NAMESPACE(rej_uniform_avx)
unsigned int rej_uniform_avx(int16_t *r, const uint8_t *buf);

#endif
