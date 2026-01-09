#ifndef AVX_MLWQ_H
#define AVX_MLWQ_H

#include "../common/structs.h"

// PKE
void avx_mlwq_keygen(mlwq_pk *pk, mlwq_sk *sk, const uint8_t *seed_A, const uint8_t *seed_d);
void avx_mlwq_encrypt(mlwq_ciphertext *ct, const mlwq_pk *pk, const uint8_t *msg, const uint8_t *seed_ct);
void avx_mlwq_decrypt(uint8_t *msg, const mlwq_sk *sk, const mlwq_ciphertext *ct);
void avx_poly_getnoise_eta1(poly *r, const uint8_t seed[32], uint8_t nonce);
void avx_polyvec_getnoise_eta1(poly_vec *r, const uint8_t seed[32], uint8_t nonce_base);

// KEM
void avx_mlwq_kem_keygen(mlwq_pk *pk, mlwq_kem_sk *sk);
void avx_mlwq_kem_encaps(mlwq_ciphertext *ct, uint8_t *ss, const mlwq_pk *pk);
int avx_mlwq_kem_decaps(uint8_t *ss, const mlwq_kem_sk *sk, const mlwq_ciphertext *ct);

#endif
