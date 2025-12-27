#pragma once
#include <vector>
#include <cstdint>
#include <memory>
#include "../common/structs.hpp"
#include "keccak4x.hpp"

struct sha3_context; 

namespace mlwq {
namespace avx2 {
    class Shake128 {
    public:
        Shake128();
        ~Shake128();
        void update(const std::vector<uint8_t>& data);
        void finalize();
        void digest(std::vector<uint8_t>& output, size_t len);
    private:
        std::unique_ptr<sha3_context> ctx;
    };

    class Shake128x4 {
    public:
        Shake128x4();
        ~Shake128x4();
        void update4(const std::vector<uint8_t>& d0, const std::vector<uint8_t>& d1, const std::vector<uint8_t>& d2, const std::vector<uint8_t>& d3);
        void finalize4();
        void digest4(std::vector<uint8_t>& out0, std::vector<uint8_t>& out1, std::vector<uint8_t>& out2, std::vector<uint8_t>& out3, size_t len);
    private:
        std::unique_ptr<Keccak4x_State> state;
        std::vector<uint8_t> seeds[4];
    };

    void xof_expand_matrix_x4(poly_matrix& A, const std::vector<uint8_t>& seed, int32_t modulus);
    void xof_expand_poly_vec(poly_vec& v, const std::vector<uint8_t>& seed, int32_t k, int32_t modulus);
}
}