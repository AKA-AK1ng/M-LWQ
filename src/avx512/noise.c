#include <string.h>
#include <immintrin.h>
#include "noise.h"
#include "fips202x8.h"
#include "align.h"
#include "../common/fips202.h"
#include "../common/params.h"

// =========================================================================
// AVX2 CBD3 implementation reused for AVX512 noise sampling
// =========================================================================
static void avx_cbd3(poly *restrict r, const uint8_t buf[3 * MLWQ_N / 4 + 8]) {
    unsigned int i;
    __m256i f0, f1, f2, f3;
    const __m256i mask249 = _mm256_set1_epi32(0x249249);
    const __m256i mask6db = _mm256_set1_epi32(0x6DB6DB);
    const __m256i mask07 = _mm256_set1_epi32(7);
    const __m256i mask70 = _mm256_set1_epi32(7 << 16);
    const __m256i mask3 = _mm256_set1_epi16(3);
    const __m256i shufbidx = _mm256_set_epi8(-1, 15, 14, 13, -1, 12, 11, 10, -1, 9, 8, 7, -1, 6, 5, 4,
                                             -1, 11, 10, 9, -1, 8, 7, 6, -1, 5, 4, 3, -1, 2, 1, 0);

    for (i = 0; i < MLWQ_N / 32; i++) {
        f0 = _mm256_loadu_si256((const __m256i *)&buf[24 * i]);
        f0 = _mm256_permute4x64_epi64(f0, 0x94);
        f0 = _mm256_shuffle_epi8(f0, shufbidx);

        f1 = _mm256_srli_epi32(f0, 1);
        f2 = _mm256_srli_epi32(f0, 2);
        f0 = _mm256_and_si256(mask249, f0);
        f1 = _mm256_and_si256(mask249, f1);
        f2 = _mm256_and_si256(mask249, f2);
        f0 = _mm256_add_epi32(f0, f1);
        f0 = _mm256_add_epi32(f0, f2);

        f1 = _mm256_srli_epi32(f0, 3);
        f0 = _mm256_add_epi32(f0, mask6db);
        f0 = _mm256_sub_epi32(f0, f1);

        f1 = _mm256_slli_epi32(f0, 10);
        f2 = _mm256_srli_epi32(f0, 12);
        f3 = _mm256_srli_epi32(f0, 2);
        f0 = _mm256_and_si256(f0, mask07);
        f1 = _mm256_and_si256(f1, mask70);
        f2 = _mm256_and_si256(f2, mask07);
        f3 = _mm256_and_si256(f3, mask70);
        f0 = _mm256_add_epi16(f0, f1);
        f1 = _mm256_add_epi16(f2, f3);
        f0 = _mm256_sub_epi16(f0, mask3);
        f1 = _mm256_sub_epi16(f1, mask3);

        f2 = _mm256_unpacklo_epi32(f0, f1);
        f3 = _mm256_unpackhi_epi32(f0, f1);

        f0 = _mm256_permute2x128_si256(f2, f3, 0x20);
        f1 = _mm256_permute2x128_si256(f2, f3, 0x31);

        _mm256_store_si256((__m256i *)&r->coeffs[32 * i + 0], f0);
        _mm256_store_si256((__m256i *)&r->coeffs[32 * i + 16], f1);
    }
}

static void avx512_poly_cbd_eta1(poly *r, const __m256i buf[MLWQ_ETA1 * MLWQ_N / 128 + 1]) {
#if MLWQ_ETA1 == 3
    avx_cbd3(r, (const uint8_t *)buf);
#else
#error "AVX512 cbd requires MLWQ_ETA1 == 3"
#endif
}

static void avx512_poly_getnoise_eta1_8x(poly *r0,
                                         poly *r1,
                                         poly *r2,
                                         poly *r3,
                                         poly *r4,
                                         poly *r5,
                                         poly *r6,
                                         poly *r7,
                                         const uint8_t seed[32],
                                         uint8_t nonce0,
                                         uint8_t nonce1,
                                         uint8_t nonce2,
                                         uint8_t nonce3,
                                         uint8_t nonce4,
                                         uint8_t nonce5,
                                         uint8_t nonce6,
                                         uint8_t nonce7) {
#define NOISE_NBLOCKS ((MLWQ_ETA1 * MLWQ_N / 4 + SHAKE256_RATE - 1) / SHAKE256_RATE)
    ALIGNED_UINT8(NOISE_NBLOCKS * SHAKE256_RATE) buf[8];
    keccakx8_state state;

    for (int i = 0; i < 8; i++) {
        memcpy(buf[i].coeffs, seed, 32);
    }

    buf[0].coeffs[32] = nonce0;
    buf[1].coeffs[32] = nonce1;
    buf[2].coeffs[32] = nonce2;
    buf[3].coeffs[32] = nonce3;
    buf[4].coeffs[32] = nonce4;
    buf[5].coeffs[32] = nonce5;
    buf[6].coeffs[32] = nonce6;
    buf[7].coeffs[32] = nonce7;

    shake256x8_absorb_once(&state,
                           buf[0].coeffs,
                           buf[1].coeffs,
                           buf[2].coeffs,
                           buf[3].coeffs,
                           buf[4].coeffs,
                           buf[5].coeffs,
                           buf[6].coeffs,
                           buf[7].coeffs,
                           33);
    shake256x8_squeezeblocks(buf[0].coeffs,
                             buf[1].coeffs,
                             buf[2].coeffs,
                             buf[3].coeffs,
                             buf[4].coeffs,
                             buf[5].coeffs,
                             buf[6].coeffs,
                             buf[7].coeffs,
                             NOISE_NBLOCKS,
                             &state);

    avx512_poly_cbd_eta1(r0, (const __m256i *)buf[0].coeffs);
    avx512_poly_cbd_eta1(r1, (const __m256i *)buf[1].coeffs);
    avx512_poly_cbd_eta1(r2, (const __m256i *)buf[2].coeffs);
    avx512_poly_cbd_eta1(r3, (const __m256i *)buf[3].coeffs);
    avx512_poly_cbd_eta1(r4, (const __m256i *)buf[4].coeffs);
    avx512_poly_cbd_eta1(r5, (const __m256i *)buf[5].coeffs);
    avx512_poly_cbd_eta1(r6, (const __m256i *)buf[6].coeffs);
    avx512_poly_cbd_eta1(r7, (const __m256i *)buf[7].coeffs);
#undef NOISE_NBLOCKS
}

void avx512_poly_getnoise_eta1(poly *r, const uint8_t seed[32], uint8_t nonce) {
    ALIGNED_UINT8(MLWQ_ETA1 * MLWQ_N / 4 + 32) buf;
    uint8_t ext_seed[33];

    memcpy(ext_seed, seed, 32);
    ext_seed[32] = nonce;
    shake256(buf.coeffs, MLWQ_ETA1 * MLWQ_N / 4, ext_seed, sizeof(ext_seed));
    avx512_poly_cbd_eta1(r, buf.vec);
}

void avx512_polyvec_getnoise_eta1(poly_vec *r, const uint8_t seed[32], uint8_t nonce_base) {
    int i = 0;
    while (i + 7 < MLWQ_K) {
        avx512_poly_getnoise_eta1_8x(&r->vec[i + 0],
                                     &r->vec[i + 1],
                                     &r->vec[i + 2],
                                     &r->vec[i + 3],
                                     &r->vec[i + 4],
                                     &r->vec[i + 5],
                                     &r->vec[i + 6],
                                     &r->vec[i + 7],
                                     seed,
                                     (uint8_t)(nonce_base + i + 0),
                                     (uint8_t)(nonce_base + i + 1),
                                     (uint8_t)(nonce_base + i + 2),
                                     (uint8_t)(nonce_base + i + 3),
                                     (uint8_t)(nonce_base + i + 4),
                                     (uint8_t)(nonce_base + i + 5),
                                     (uint8_t)(nonce_base + i + 6),
                                     (uint8_t)(nonce_base + i + 7));
        i += 8;
    }
    if (i < MLWQ_K) {
        poly tmp[7];
        poly *p0 = &r->vec[i + 0];
        poly *p1 = (i + 1 < MLWQ_K) ? &r->vec[i + 1] : &tmp[0];
        poly *p2 = (i + 2 < MLWQ_K) ? &r->vec[i + 2] : &tmp[1];
        poly *p3 = (i + 3 < MLWQ_K) ? &r->vec[i + 3] : &tmp[2];
        poly *p4 = (i + 4 < MLWQ_K) ? &r->vec[i + 4] : &tmp[3];
        poly *p5 = (i + 5 < MLWQ_K) ? &r->vec[i + 5] : &tmp[4];
        poly *p6 = (i + 6 < MLWQ_K) ? &r->vec[i + 6] : &tmp[5];
        poly *p7 = (i + 7 < MLWQ_K) ? &r->vec[i + 7] : &tmp[6];
        avx512_poly_getnoise_eta1_8x(p0,
                                     p1,
                                     p2,
                                     p3,
                                     p4,
                                     p5,
                                     p6,
                                     p7,
                                     seed,
                                     (uint8_t)(nonce_base + i + 0),
                                     (uint8_t)(nonce_base + i + 1),
                                     (uint8_t)(nonce_base + i + 2),
                                     (uint8_t)(nonce_base + i + 3),
                                     (uint8_t)(nonce_base + i + 4),
                                     (uint8_t)(nonce_base + i + 5),
                                     (uint8_t)(nonce_base + i + 6),
                                     (uint8_t)(nonce_base + i + 7));
    }
}
