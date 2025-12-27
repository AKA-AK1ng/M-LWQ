#pragma once
#include <vector>
#include <cstdint>
#include "params.hpp"

namespace mlwq {
    using poly = std::vector<int16_t>;
    using poly_vec = std::vector<poly>;
    using poly_matrix = std::vector<poly_vec>;

    // 性能统计
    struct MlwqProfiling {
        uint64_t kg_gen_A = 0;      
        uint64_t kg_sample_s = 0;   
        uint64_t kg_gen_d = 0;      
        uint64_t kg_arith_as = 0;   
        uint64_t kg_quant = 0;      

        uint64_t enc_gen_A = 0;     
        uint64_t enc_sample_r = 0;  
        uint64_t enc_gen_d = 0;     
        uint64_t enc_arith_u = 0;   
        uint64_t enc_arith_v = 0;   
        uint64_t enc_quant = 0;     

        uint64_t dec_dequant = 0;
        uint64_t dec_mul_sub = 0; 
        uint64_t dec_decode = 0;    

        void reset() { *this = MlwqProfiling(); }
    };

    // PKE 公钥
    struct mlwq_pk {
        std::vector<uint8_t> seed_A;
        std::vector<uint8_t> seed_d_pk;
        poly_vec b_q;
    };

    // PKE 私钥
    struct mlwq_sk {
        poly_vec s;
    };

    // KEM 私钥 (包含隐式拒绝所需的 z 和 h_pk)
    struct mlwq_kem_sk {
        mlwq_sk pke_sk;             
        mlwq_pk pk;                 
        std::vector<uint8_t> h_pk;  
        std::vector<uint8_t> z;     
    };

    // 密文
    struct mlwq_ciphertext {
        poly_vec u;
        poly v;
    };
}