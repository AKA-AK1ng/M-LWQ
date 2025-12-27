#include "xof.hpp"
#include <stdexcept>

namespace mlwq {
namespace ref {

Shake128::Shake128() { ctx = std::make_unique<sha3_context>(); shake128_init(ctx.get()); }
Shake128::~Shake128() = default;
void Shake128::update(const std::vector<uint8_t>& data) { shake128_update(ctx.get(), data.data(), data.size()); }
void Shake128::finalize() { shake128_xof(ctx.get()); }
void Shake128::digest(std::vector<uint8_t>& output, size_t len) { output.resize(len); shake128_out(ctx.get(), output.data(), len); }

// [Fix] 实现 reset
void Shake128::reset() {
    shake128_init(ctx.get());
}

static void bytes_to_poly(poly& p, const std::vector<uint8_t>& bytes, int32_t modulus) {
    p.resize(params::N);
    for (size_t i = 0; i < params::N; ++i) {
        uint16_t val = static_cast<uint16_t>(bytes[2*i]) | (static_cast<uint16_t>(bytes[2*i + 1]) << 8);
        p[i] = static_cast<int16_t>(val % modulus);
    }
}

void xof_expand_poly_vec(poly_vec& v, const std::vector<uint8_t>& seed, int32_t k, int32_t modulus) {
    v.resize(k);
    Shake128 shake; shake.update(seed); shake.finalize();
    std::vector<uint8_t> raw;
    for (int i = 0; i < k; ++i) { shake.digest(raw, 2 * params::N); bytes_to_poly(v[i], raw, modulus); }
}

void xof_expand_matrix(poly_matrix& A, const std::vector<uint8_t>& seed, int32_t modulus) {
    A.resize(params::K, poly_vec(params::K));
    Shake128 shake;
    for (int i = 0; i < params::K; ++i) {
        for (int j = 0; j < params::K; ++j) {
            std::vector<uint8_t> s = seed; s.push_back(i); s.push_back(j);
            shake.update(s); shake.finalize(); 
            std::vector<uint8_t> raw; shake.digest(raw, 2 * params::N);
            bytes_to_poly(A[i][j], raw, modulus);
            
            // [Fix] 使用 reset() 替代赋值
            shake.reset(); 
        }
    }
}
}
}