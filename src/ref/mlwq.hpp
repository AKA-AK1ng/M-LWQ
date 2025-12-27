#pragma once
#include "../common/structs.hpp"
#include <utility>

namespace mlwq {
namespace ref {
    std::pair<mlwq_pk, mlwq_sk> mlwq_keygen(const std::vector<uint8_t>& seed_A, const std::vector<uint8_t>& seed_d_pk, MlwqProfiling* stats = nullptr);
    mlwq_ciphertext mlwq_encrypt(const mlwq_pk& pk, const poly& m_poly, const std::vector<uint8_t>& seed_ct, MlwqProfiling* stats = nullptr);
    poly mlwq_decrypt(const mlwq_sk& sk, const mlwq_ciphertext& ct, MlwqProfiling* stats = nullptr);

    std::pair<mlwq_pk, mlwq_kem_sk> mlwq_kem_keygen(MlwqProfiling* stats = nullptr);
    std::pair<mlwq_ciphertext, std::vector<uint8_t>> mlwq_kem_encaps(const mlwq_pk& pk, MlwqProfiling* stats = nullptr);
    std::vector<uint8_t> mlwq_kem_decaps(const mlwq_kem_sk& sk, const mlwq_ciphertext& ct, MlwqProfiling* stats = nullptr);
    
    // [Deleted] check_poly_eq declaration removed to avoid conflict
}
}