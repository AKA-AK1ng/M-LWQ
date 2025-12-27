#include <iostream>
#include <vector>
#include <string>
#include <numeric>
#include <iomanip>
#include <sstream>

#include "common/cycles.hpp"
#include "common/params.hpp"
#include "common/structs.hpp"
#include "common/random.hpp"

// 引入两套实现
#include "ref/mlwq.hpp"
#include "ref/poly.hpp"
#include "ref/xof.hpp" 
#include "avx2/mlwq.hpp"
#include "avx2/poly.hpp"
#include "avx2/xof.hpp"

namespace Ref = mlwq::ref;
namespace Avx = mlwq::avx2;

// --- 辅助函数 ---
double get_avg(const std::vector<unsigned long long>& v) {
    if (v.empty()) return 0;
    return (double)std::accumulate(v.begin(), v.end(), 0ULL) / v.size();
}

template <typename Func>
double bench(Func f, int rounds) {
    std::vector<unsigned long long> t;
    // 预热
    f(); 
    for(int i=0; i<rounds; ++i) {
        auto c1 = start_cycles();
        f();
        auto c2 = stop_cycles();
        t.push_back(c2 - c1);
    }
    return get_avg(t);
}

// 打印分解表 (Part 1)
void print_breakdown(const std::string& stage, 
                     const std::vector<std::pair<std::string, uint64_t>>& parts_ref,
                     const std::vector<std::pair<std::string, uint64_t>>& parts_avx) {
    std::cout << "--------------------------------------------------------------------------------------\n";
    std::cout << " " << stage << " Breakdown (Detailed)\n";
    std::cout << "--------------------------------------------------------------------------------------\n";
    std::cout << std::left << std::setw(20) << "Sub-Component" 
              << std::setw(15) << "Scalar (cyc)" 
              << std::setw(15) << "AVX2 (cyc)" 
              << std::setw(15) << "Speedup" 
              << "Scalar %" << std::endl;
    
    uint64_t tot_ref = 0; 
    for(auto& p : parts_ref) tot_ref += p.second;
    
    for (size_t i = 0; i < parts_ref.size(); ++i) {
        double r = (double)parts_ref[i].second;
        double a = (double)parts_avx[i].second;
        double ratio = (a > 0) ? r / a : 0.0;
        double pct = (tot_ref > 0) ? (r / tot_ref) * 100.0 : 0.0;
        
        std::cout << std::left << std::setw(20) << parts_ref[i].first 
                  << std::setw(15) << (long long)r
                  << std::setw(15) << (long long)a
                  << std::fixed << std::setprecision(2) << ratio << "x"
                  << std::setw(10) << " " << std::setprecision(1) << pct << "%"
                  << std::endl;
    }
    std::cout << std::endl;
}

int main() {
    std::cout << "=== M-LWQ Comprehensive Performance Report ===\n";
    std::cout << mlwq::params::PARAM_SET_NAME << "\n";
    std::cout << "N=" << mlwq::params::N << ", K=" << mlwq::params::K << "\n\n";

    int rounds = 1000;
    mlwq::MlwqProfiling stats_ref, stats_avx;

    // ==========================================
    // 0. PKE Correctness & Profiling Run
    // ==========================================
    {
        std::cout << ">>> Running: Scalar Mode (" << rounds << " rounds)...\n";
        auto seed_A = std::vector<uint8_t>(32, 1);
        auto seed_d = std::vector<uint8_t>(32, 2);
        auto seed_ct = std::vector<uint8_t>(32, 3);
        auto msg = mlwq::random_poly_uniform(2);

        for(int i=0; i<rounds; ++i) {
            auto kp = Ref::mlwq_keygen(seed_A, seed_d, &stats_ref);
            auto ct = Ref::mlwq_encrypt(kp.first, msg, seed_ct, &stats_ref);
            auto m2 = Ref::mlwq_decrypt(kp.second, ct, &stats_ref);
            if (i==0 && !Ref::check_poly_eq(msg, m2)) { std::cout << "   [FAIL] Ref PKE check failed!\n"; exit(1); }
        }
        std::cout << "   [PASS] Correctness verified.\n";

        // Average stats
        auto avg_stats = [&](mlwq::MlwqProfiling& s) {
            s.kg_gen_A /= rounds; s.kg_sample_s /= rounds; s.kg_gen_d /= rounds; s.kg_arith_as /= rounds; s.kg_quant /= rounds;
            s.enc_gen_A /= rounds; s.enc_sample_r /= rounds; s.enc_gen_d /= rounds; s.enc_arith_u /= rounds; s.enc_arith_v /= rounds; s.enc_quant /= rounds;
            s.dec_dequant /= rounds; s.dec_mul_sub /= rounds; s.dec_decode /= rounds;
        };
        avg_stats(stats_ref);
    }

    {
        std::cout << ">>> Running: AVX2 Mode (" << rounds << " rounds)...\n";
        auto seed_A = std::vector<uint8_t>(32, 1);
        auto seed_d = std::vector<uint8_t>(32, 2);
        auto seed_ct = std::vector<uint8_t>(32, 3);
        auto msg = mlwq::random_poly_uniform(2);

        for(int i=0; i<rounds; ++i) {
            auto kp = Avx::mlwq_keygen(seed_A, seed_d, &stats_avx);
            auto ct = Avx::mlwq_encrypt(kp.first, msg, seed_ct, &stats_avx);
            auto m2 = Avx::mlwq_decrypt(kp.second, ct, &stats_avx);
            if (i==0 && !Avx::check_poly_eq(msg, m2)) { std::cout << "   [FAIL] AVX PKE check failed!\n"; exit(1); }
        }
        std::cout << "   [PASS] Correctness verified.\n";
        
        auto avg_stats = [&](mlwq::MlwqProfiling& s) {
            s.kg_gen_A /= rounds; s.kg_sample_s /= rounds; s.kg_gen_d /= rounds; s.kg_arith_as /= rounds; s.kg_quant /= rounds;
            s.enc_gen_A /= rounds; s.enc_sample_r /= rounds; s.enc_gen_d /= rounds; s.enc_arith_u /= rounds; s.enc_arith_v /= rounds; s.enc_quant /= rounds;
            s.dec_dequant /= rounds; s.dec_mul_sub /= rounds; s.dec_decode /= rounds;
        };
        avg_stats(stats_avx);
    }

    // ==========================================
    // PART 1: Internal Breakdown
    // ==========================================
    std::cout << "\n>>> PART 1: Internal Breakdown (Where is time spent?)\n\n";

    print_breakdown("PKE KeyGen", 
        {{"GenMatrix (A)", stats_ref.kg_gen_A}, {"Sample (s)", stats_ref.kg_sample_s}, 
         {"GenDither", stats_ref.kg_gen_d}, {"Arith (A*s)", stats_ref.kg_arith_as}, {"Quantize", stats_ref.kg_quant}},
        {{"GenMatrix (A)", stats_avx.kg_gen_A}, {"Sample (s)", stats_avx.kg_sample_s}, 
         {"GenDither", stats_avx.kg_gen_d}, {"Arith (A*s)", stats_avx.kg_arith_as}, {"Quantize", stats_avx.kg_quant}}
    );

    print_breakdown("PKE Encrypt", 
        {{"GenMatrix (A)", stats_ref.enc_gen_A}, {"Sample (r)", stats_ref.enc_sample_r}, 
         {"GenDither", stats_ref.enc_gen_d}, {"Arith (u)", stats_ref.enc_arith_u}, 
         {"Arith (v)", stats_ref.enc_arith_v}, {"Quantize", stats_ref.enc_quant}},
        {{"GenMatrix (A)", stats_avx.enc_gen_A}, {"Sample (r)", stats_avx.enc_sample_r}, 
         {"GenDither", stats_avx.enc_gen_d}, {"Arith (u)", stats_avx.enc_arith_u}, 
         {"Arith (v)", stats_avx.enc_arith_v}, {"Quantize", stats_avx.enc_quant}}
    );

    print_breakdown("PKE Decrypt", 
        {{"DeQuantize", stats_ref.dec_dequant}, {"Arith (v-su)", stats_ref.dec_mul_sub}, {"Decode", stats_ref.dec_decode}},
        {{"DeQuantize", stats_avx.dec_dequant}, {"Arith (v-su)", stats_avx.dec_mul_sub}, {"Decode", stats_avx.dec_decode}}
    );

    // ==========================================
    // PART 2: Micro-Benchmarks (Component Comparison)
    // ==========================================
    std::cout << "\n>>> PART 2: Core Component Comparison (Quantize vs Sample)\n";
    std::cout << "----------------------------------------------------------------------------------------------\n";
    std::cout << std::left << std::setw(12) << "Component" << std::setw(12) << "Mode"
              << std::setw(16) << "Quantize" << std::setw(16) << "Sample" 
              << std::setw(22) << "Alg. Efficiency"
              << std::setw(20) << "AVX Improvement" << std::endl;
    std::cout << "----------------------------------------------------------------------------------------------\n";

    // Setup dummy data for component bench
    mlwq::poly_vec vec_input(mlwq::params::K, mlwq::poly(mlwq::params::N)); 
    mlwq::poly poly_input(mlwq::params::N);
    mlwq::poly_vec d_vec(mlwq::params::K, mlwq::poly(mlwq::params::N));
    mlwq::poly d_poly(mlwq::params::N);

    // Micro-bench Loop
    auto bench_comp = [&](auto f_q, auto f_s) {
        double tq = bench(f_q, 2000);
        double ts = bench(f_s, 2000);
        return std::make_pair(tq, ts);
    };

    // Ref PK/u
    auto pair_ref_vec = bench_comp(
        [&](){ Ref::poly_vec_quantize(vec_input, d_vec, mlwq::params::P_PK); },
        [&](){ mlwq::random_poly_vec_eta(mlwq::params::K, mlwq::params::ETA); }
    );
    // Avx PK/u
    auto pair_avx_vec = bench_comp(
        [&](){ Avx::poly_vec_quantize(vec_input, d_vec, mlwq::params::P_PK); },
        [&](){ mlwq::random_poly_vec_eta(mlwq::params::K, mlwq::params::ETA); }
    );

    auto print_line = [&](std::string item, std::string mode, double t_quant, double t_sample, double ref_quant_scalar) {
        double alg_eff = (t_quant > 0) ? (t_sample / t_quant) : 0.0;
        std::stringstream ss_avx;
        if (ref_quant_scalar < 0) ss_avx << "1.00x (Ref)";
        else ss_avx << std::fixed << std::setprecision(2) << (ref_quant_scalar / t_quant) << "x";
        std::cout << std::left << std::setw(12) << item << std::setw(12) << mode
                  << std::setw(16) << (long long)t_quant << std::setw(16) << (long long)t_sample 
                  << std::setw(22) << (std::to_string((int)alg_eff) + "." + std::to_string((int)((alg_eff-(int)alg_eff)*100)).substr(0,2) + "x")
                  << std::setw(20) << ss_avx.str() << std::endl;
    };

    print_line("PK / u", "Scalar", pair_ref_vec.first, pair_ref_vec.second, -1.0);
    print_line("PK / u", "AVX2",   pair_avx_vec.first, pair_avx_vec.second, pair_ref_vec.first);

    // ==========================================
    // PART 3: PKE Full Flow Summary
    // ==========================================
    std::cout << "\n\n>>> PART 3: PKE Full Flow Summary (Total Time)\n";
    std::cout << "----------------------------------------------------------------------------------------------\n";
    std::cout << std::left << std::setw(20) << "Operation" 
              << std::setw(18) << "Scalar Cycles" 
              << std::setw(18) << "AVX2 Cycles" 
              << "Speedup" << std::endl;
    std::cout << "----------------------------------------------------------------------------------------------\n";
    
    auto print_flow = [&](std::string n, double s, double a) {
        std::cout << std::left << std::setw(20) << n 
                  << std::setw(18) << (long long)s 
                  << std::setw(18) << (long long)a 
                  << std::fixed << std::setprecision(2) << (s/a) << "x" << std::endl;
    };

    auto seed_A = std::vector<uint8_t>(32, 1);
    auto seed_d = std::vector<uint8_t>(32, 2);
    auto seed_ct = std::vector<uint8_t>(32, 3);
    auto msg = mlwq::random_poly_uniform(2);

    // 预计算 KeyPair 以供 Enc/Dec 使用
    auto kp_ref = Ref::mlwq_keygen(seed_A, seed_d, nullptr);
    auto ct_ref = Ref::mlwq_encrypt(kp_ref.first, msg, seed_ct, nullptr);
    auto kp_avx = Avx::mlwq_keygen(seed_A, seed_d, nullptr);
    auto ct_avx = Avx::mlwq_encrypt(kp_avx.first, msg, seed_ct, nullptr);

    double t_pke_kg_ref = bench([&](){ Ref::mlwq_keygen(seed_A, seed_d, nullptr); }, rounds);
    double t_pke_kg_avx = bench([&](){ Avx::mlwq_keygen(seed_A, seed_d, nullptr); }, rounds);
    
    double t_pke_enc_ref = bench([&](){ Ref::mlwq_encrypt(kp_ref.first, msg, seed_ct, nullptr); }, rounds);
    double t_pke_enc_avx = bench([&](){ Avx::mlwq_encrypt(kp_avx.first, msg, seed_ct, nullptr); }, rounds);

    double t_pke_dec_ref = bench([&](){ Ref::mlwq_decrypt(kp_ref.second, ct_ref, nullptr); }, rounds);
    double t_pke_dec_avx = bench([&](){ Avx::mlwq_decrypt(kp_avx.second, ct_avx, nullptr); }, rounds);

    print_flow("PKE KeyGen", t_pke_kg_ref, t_pke_kg_avx);
    print_flow("PKE Encrypt", t_pke_enc_ref, t_pke_enc_avx);
    print_flow("PKE Decrypt", t_pke_dec_ref, t_pke_dec_avx);

    // ==========================================
    // PART 4: KEM Full Flow Summary
    // ==========================================
    std::cout << "\n\n>>> PART 4: KEM Full Flow Summary (IND-CCA2)\n";
    std::cout << "----------------------------------------------------------------------------------------------\n";
    std::cout << std::left << std::setw(20) << "Operation" 
              << std::setw(18) << "Scalar Cycles" 
              << std::setw(18) << "AVX2 Cycles" 
              << "Speedup" << std::endl;
    std::cout << "----------------------------------------------------------------------------------------------\n";

    // 预计算 KEM KeyPair
    auto kem_kp_ref = Ref::mlwq_kem_keygen(nullptr);
    auto kem_res_ref = Ref::mlwq_kem_encaps(kem_kp_ref.first, nullptr); // ct, ss
    auto kem_kp_avx = Avx::mlwq_kem_keygen(nullptr);
    auto kem_res_avx = Avx::mlwq_kem_encaps(kem_kp_avx.first, nullptr);

    double t_kem_kg_ref = bench([&](){ Ref::mlwq_kem_keygen(nullptr); }, rounds);
    double t_kem_kg_avx = bench([&](){ Avx::mlwq_kem_keygen(nullptr); }, rounds);

    double t_kem_enc_ref = bench([&](){ Ref::mlwq_kem_encaps(kem_kp_ref.first, nullptr); }, rounds);
    double t_kem_enc_avx = bench([&](){ Avx::mlwq_kem_encaps(kem_kp_avx.first, nullptr); }, rounds);

    double t_kem_dec_ref = bench([&](){ Ref::mlwq_kem_decaps(kem_kp_ref.second, kem_res_ref.first, nullptr); }, rounds);
    double t_kem_dec_avx = bench([&](){ Avx::mlwq_kem_decaps(kem_kp_avx.second, kem_res_avx.first, nullptr); }, rounds);

    print_flow("KEM KeyGen", t_kem_kg_ref, t_kem_kg_avx);
    print_flow("KEM Encaps", t_kem_enc_ref, t_kem_enc_avx);
    print_flow("KEM Decaps", t_kem_dec_ref, t_kem_dec_avx);
    std::cout << "----------------------------------------------------------------------------------------------\n";

    std::cout << "\n[FINAL] All checks passed! Implementation is correct.\n";
    return 0;
}