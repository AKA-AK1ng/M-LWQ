#include "mlwq.hpp"
#include "xof.hpp"
#include "poly.hpp"
#include "ntt.hpp"
#include "../common/random.hpp"
#include "../common/cycles.hpp"
#include "../common/sha3.hpp" 

#define PROFILE_START(t) uint64_t t = start_cycles();
#define PROFILE_END(stats, field, t) if (stats) { stats->field += stop_cycles() - t; }

namespace mlwq {
namespace ref {

// 辅助函数：确定性采样 r
poly_vec sample_deterministic_r(int32_t k, int32_t eta, const std::vector<uint8_t>& seed) {
    poly_vec pv(k);
    Shake128 shake;
    shake.update(seed);
    shake.finalize();
    std::vector<uint8_t> buf(params::N * 2);
    int32_t range = 2 * eta + 1;
    for (int i = 0; i < k; ++i) {
        pv[i].resize(params::N);
        int count = 0;
        while (count < params::N) {
            shake.digest(buf, buf.size());
            for (size_t j = 0; j < buf.size() && count < params::N; ++j) {
                uint32_t val = buf[j];
                if (val < 255 - (255 % range)) {
                    int32_t sample = (val % range) - eta;
                    pv[i][count++] = (int16_t)sample;
                }
            }
        }
    }
    return pv;
}

// 辅助哈希函数
std::vector<uint8_t> serialize_pk(const mlwq_pk& pk) {
    std::vector<uint8_t> buf = pk.seed_A;
    buf.insert(buf.end(), pk.seed_d_pk.begin(), pk.seed_d_pk.end());
    return buf;
}
bool ct_equal(const mlwq_ciphertext& a, const mlwq_ciphertext& b) {
    if (a.u.size() != b.u.size()) return false;
    for(size_t i=0; i<a.u.size(); ++i) {
        for(size_t j=0; j<params::N; ++j) if (a.u[i][j] != b.u[i][j]) return false;
    }
    for(size_t i=0; i<params::N; ++i) if (a.v[i] != b.v[i]) return false;
    return true;
}
std::vector<uint8_t> hash_h(const std::vector<uint8_t>& in) {
    std::vector<uint8_t> out(32);
    Shake128 shake; shake.update(in); shake.finalize(); shake.digest(out, 32);
    return out;
}
std::pair<std::vector<uint8_t>, std::vector<uint8_t>> hash_g(const std::vector<uint8_t>& in) {
    std::vector<uint8_t> out(64);
    Shake128 shake; shake.update(in); shake.finalize(); shake.digest(out, 64);
    std::vector<uint8_t> k(out.begin(), out.begin() + 32);
    std::vector<uint8_t> r(out.begin() + 32, out.end());
    return {k, r};
}

// --- PKE ---
std::pair<mlwq_pk, mlwq_sk> mlwq_keygen(const std::vector<uint8_t>& seed_A, const std::vector<uint8_t>& seed_d_pk, MlwqProfiling* stats) {
    PROFILE_START(t1);
    poly_matrix A;
    xof_expand_matrix(A, seed_A, params::Q);
    PROFILE_END(stats, kg_gen_A, t1);
    
    PROFILE_START(t2);
    mlwq_sk sk;
    sk.s = random_poly_vec_eta(params::K, params::ETA);
    PROFILE_END(stats, kg_sample_s, t2);

    PROFILE_START(t3);
    poly_vec d_pk;
    xof_expand_poly_vec(d_pk, seed_d_pk, params::K, params::Q / params::P_PK);
    PROFILE_END(stats, kg_gen_d, t3);

    PROFILE_START(t4);
    poly_vec As = poly_matrix_vec_mul(A, sk.s);
    PROFILE_END(stats, kg_arith_as, t4);

    PROFILE_START(t5);
    mlwq_pk pk;
    pk.seed_A = seed_A;
    pk.seed_d_pk = seed_d_pk;
    pk.b_q = poly_vec_quantize(As, d_pk, params::P_PK);
    PROFILE_END(stats, kg_quant, t5);

    return {pk, sk};
}

mlwq_ciphertext mlwq_encrypt(const mlwq_pk& pk, const poly& m_poly, const std::vector<uint8_t>& seed_ct, MlwqProfiling* stats) {
    PROFILE_START(t1);
    poly_matrix A;
    xof_expand_matrix(A, pk.seed_A, params::Q);
    PROFILE_END(stats, enc_gen_A, t1);

    std::vector<uint8_t> seed_r = seed_ct; seed_r.push_back(0x00);
    std::vector<uint8_t> seed_d = seed_ct; seed_d.push_back(0x01);

    PROFILE_START(t2);
    poly_vec r = sample_deterministic_r(params::K, params::ETA, seed_r);
    PROFILE_END(stats, enc_sample_r, t2);

    PROFILE_START(t3);
    std::vector<uint8_t> seed_d_u = seed_d; seed_d_u.push_back(0x00);
    poly_vec d_u;
    xof_expand_poly_vec(d_u, seed_d_u, params::K, params::Q / params::P_U);
    std::vector<uint8_t> seed_d_v = seed_d; seed_d_v.push_back(0x01);
    poly_vec d_v_vec;
    xof_expand_poly_vec(d_v_vec, seed_d_v, 1, params::Q / params::P_V);
    PROFILE_END(stats, enc_gen_d, t3);
    
    PROFILE_START(t4);
    poly_matrix A_t = poly_matrix_transpose(A);
    poly_vec A_t_r = poly_matrix_vec_mul(A_t, r);
    PROFILE_END(stats, enc_arith_u, t4);

    PROFILE_START(t5);
    poly_vec b_q_deq = poly_vec_dequantize(pk.b_q, params::P_PK);
    poly b_t_r = poly_vec_transpose_mul(b_q_deq, r);
    poly m_enc = poly_message_encode(m_poly);
    poly v_val = poly_add(b_t_r, m_enc);
    PROFILE_END(stats, enc_arith_v, t5);

    PROFILE_START(t6);
    mlwq_ciphertext ct;
    ct.u = poly_vec_quantize(A_t_r, d_u, params::P_U);
    ct.v = poly_quantize(v_val, d_v_vec[0], params::P_V);
    PROFILE_END(stats, enc_quant, t6);

    return ct;
}

poly mlwq_decrypt(const mlwq_sk& sk, const mlwq_ciphertext& ct, MlwqProfiling* stats) {
    PROFILE_START(t1);
    poly_vec u_deq = poly_vec_dequantize(ct.u, params::P_U);
    poly v_deq = poly_dequantize(ct.v, params::P_V);
    PROFILE_END(stats, dec_dequant, t1);

    PROFILE_START(t2);
    poly s_t_u = poly_vec_transpose_mul(sk.s, u_deq);
    poly val = poly_sub(v_deq, s_t_u);
    PROFILE_END(stats, dec_mul_sub, t2);

    PROFILE_START(t3);
    poly m = poly_message_decode(val);
    PROFILE_END(stats, dec_decode, t3);

    return m;
}

// --- KEM ---
std::pair<mlwq_pk, mlwq_kem_sk> mlwq_kem_keygen(MlwqProfiling* stats) {
    poly rand_p = random_poly_uniform(256);
    std::vector<uint8_t> seed_A(32), seed_d(32);
    for(int i=0; i<32; ++i) { seed_A[i] = (uint8_t)rand_p[i]; seed_d[i] = (uint8_t)rand_p[i+32]; }
    auto kp = mlwq_keygen(seed_A, seed_d, stats);
    mlwq_kem_sk sk_kem;
    sk_kem.pke_sk = kp.second;
    sk_kem.pk = kp.first;
    sk_kem.h_pk = hash_h(serialize_pk(kp.first));
    sk_kem.z.resize(32);
    poly rand_z = random_poly_uniform(256);
    for(int i=0; i<32; ++i) sk_kem.z[i] = (uint8_t)rand_z[i];
    return {kp.first, sk_kem};
}

std::pair<mlwq_ciphertext, std::vector<uint8_t>> mlwq_kem_encaps(const mlwq_pk& pk, MlwqProfiling* stats) {
    std::vector<uint8_t> m_bytes(32);
    poly rand_m = random_poly_uniform(256);
    for(int i=0; i<32; ++i) m_bytes[i] = (uint8_t)rand_m[i];
    auto h_pk = hash_h(serialize_pk(pk));
    std::vector<uint8_t> buf = m_bytes;
    buf.insert(buf.end(), h_pk.begin(), h_pk.end());
    auto g_out = hash_g(buf);
    poly m_poly(params::N);
    for(int i=0; i<params::N/8; ++i) for(int j=0; j<8; ++j) m_poly[8*i+j] = (m_bytes[i] >> j) & 1;
    mlwq_ciphertext ct = mlwq_encrypt(pk, m_poly, g_out.second, stats);
    return {ct, g_out.first};
}

std::vector<uint8_t> mlwq_kem_decaps(const mlwq_kem_sk& sk, const mlwq_ciphertext& ct, MlwqProfiling* stats) {
    poly m_prime_poly = mlwq_decrypt(sk.pke_sk, ct, stats);
    std::vector<uint8_t> m_prime_bytes(32, 0);
    for(int i=0; i<params::N/8; ++i) for(int j=0; j<8; ++j) if(m_prime_poly[8*i+j]) m_prime_bytes[i] |= (1<<j);
    std::vector<uint8_t> buf = m_prime_bytes;
    buf.insert(buf.end(), sk.h_pk.begin(), sk.h_pk.end());
    auto g_out = hash_g(buf);
    mlwq_ciphertext ct_prime = mlwq_encrypt(sk.pk, m_prime_poly, g_out.second, stats);
    if (ct_equal(ct, ct_prime)) return g_out.first;
    else return hash_h(sk.z);
}

// [Deleted] check_poly_eq definition removed
}
}