#pragma once
#include <vector>
#include <cstdint>
#include <memory>
#include "../common/sha3.hpp"
#include "../common/structs.hpp"

namespace mlwq {
namespace ref {
    class Shake128 {
    public:
        Shake128();
        ~Shake128();
        void update(const std::vector<uint8_t>& data);
        void finalize();
        void digest(std::vector<uint8_t>& output, size_t len);
        
        // [Fix] 添加 reset 方法，替代赋值操作
        void reset();
    private:
        std::unique_ptr<sha3_context> ctx;
    };

    void xof_expand_matrix(poly_matrix& A, const std::vector<uint8_t>& seed, int32_t modulus);
    void xof_expand_poly_vec(poly_vec& v, const std::vector<uint8_t>& seed, int32_t k, int32_t modulus);
}
}