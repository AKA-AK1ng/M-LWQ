#include "xof.hpp"
#include "../common/sha3.hpp"
#include "keccak4x.hpp"
#include <stdexcept>
#include <vector>

namespace mlwq {
namespace avx2 {

// ==========================================
// Shake128 (Scalar) - 复用逻辑
// ==========================================
Shake128::Shake128() { ctx = std::make_unique<sha3_context>(); shake128_init(ctx.get()); }
Shake128::~Shake128() = default;
void Shake128::update(const std::vector<uint8_t>& data) { shake128_update(ctx.get(), data.data(), data.size()); }
void Shake128::finalize() { shake128_xof(ctx.get()); }
void Shake128::digest(std::vector<uint8_t>& output, size_t len) { output.resize(len); shake128_out(ctx.get(), output.data(), len); }

// ==========================================
// Shake128x4 (AVX2) - 4路并行
// ==========================================
Shake128x4::Shake128x4() { state = std::make_unique<Keccak4x_State>(); }
Shake128x4::~Shake128x4() = default;

void Shake128x4::update4(const std::vector<uint8_t>& d0, const std::vector<uint8_t>& d1, const std::vector<uint8_t>& d2, const std::vector<uint8_t>& d3) {
    seeds[0]=d0; seeds[1]=d1; seeds[2]=d2; seeds[3]=d3;
}

void Shake128x4::finalize4() {
    shake128x4_absorb_once(state.get(), seeds[0].data(), seeds[1].data(), seeds[2].data(), seeds[3].data(), seeds[0].size());
}

void Shake128x4::digest4(std::vector<uint8_t>& out0, std::vector<uint8_t>& out1, std::vector<uint8_t>& out2, std::vector<uint8_t>& out3, size_t len) {
    out0.resize(len); out1.resize(len); out2.resize(len); out3.resize(len);
    int nblocks = (len + 167) / 168; 
    std::vector<uint8_t> buf0(nblocks*168), buf1(nblocks*168), buf2(nblocks*168), buf3(nblocks*168);
    shake128x4_squeezeblocks(buf0.data(), buf1.data(), buf2.data(), buf3.data(), nblocks, state.get());
    out0.assign(buf0.begin(), buf0.begin()+len);
    out1.assign(buf1.begin(), buf1.begin()+len);
    out2.assign(buf2.begin(), buf2.begin()+len);
    out3.assign(buf3.begin(), buf3.begin()+len);
}

// 辅助: 字节转多项式
static void bytes_to_poly(poly& p, const std::vector<uint8_t>& bytes, int32_t modulus) {
    p.resize(params::N);
    for (size_t i = 0; i < params::N; ++i) {
        uint16_t val = static_cast<uint16_t>(bytes[2*i]) | (static_cast<uint16_t>(bytes[2*i + 1]) << 8);
        p[i] = static_cast<int16_t>(val % modulus);
    }
}

// 标量扩展向量 (未改动)
void xof_expand_poly_vec(poly_vec& v, const std::vector<uint8_t>& seed, int32_t k, int32_t modulus) {
    v.resize(k);
    Shake128 shake; shake.update(seed); shake.finalize();
    std::vector<uint8_t> raw;
    for (int i = 0; i < k; ++i) { shake.digest(raw, 2 * params::N); bytes_to_poly(v[i], raw, modulus); }
}

// [核心修复] 通用 AVX2 矩阵扩展
// 支持任意 K 值 (2, 3, 4)，通过循环分批处理 (4个一组)
void xof_expand_matrix_x4(poly_matrix& A, const std::vector<uint8_t>& seed, int32_t modulus) {
    // 1. 初始化矩阵大小
    A.resize(params::K);
    for(int i=0; i<params::K; ++i) {
        A[i].resize(params::K);
    }

    int total_polys = params::K * params::K;
    
    // 2. 每次处理 4 个多项式 (4-way AVX2)
    for (int start = 0; start < total_polys; start += 4) {
        std::vector<uint8_t> s[4]; // 4 个并行种子
        int coords[4][2];          // 记录每个种子对应的 (行, 列)

        // 准备 4 个种子
        for(int lane = 0; lane < 4; ++lane) {
            int idx = start + lane;
            if(idx < total_polys) {
                // 计算当前多项式在矩阵中的位置
                int r = idx / params::K;
                int c = idx % params::K;
                coords[lane][0] = r;
                coords[lane][1] = c;

                // 构造种子: seed || i || j
                s[lane] = seed;
                s[lane].push_back((uint8_t)r);
                s[lane].push_back((uint8_t)c);
            } else {
                // 填充空闲通道 (Padding)，防止空指针崩溃
                // 使用任意合法种子即可，结果会被丢弃
                s[lane] = seed; 
                s[lane].push_back(0); s[lane].push_back(0);
            }
        }

        // 并行运行 SHAKE
        Shake128x4 shake4; 
        shake4.update4(s[0], s[1], s[2], s[3]); 
        shake4.finalize4();
        
        // 并行挤出数据
        std::vector<uint8_t> out[4];
        shake4.digest4(out[0], out[1], out[2], out[3], 2 * params::N);
        
        // 将数据填充回矩阵 A
        for(int lane = 0; lane < 4; ++lane) {
            int idx = start + lane;
            if(idx < total_polys) {
                int r = coords[lane][0];
                int c = coords[lane][1];
                bytes_to_poly(A[r][c], out[lane], modulus);
            }
        }
    }
}

} // namespace avx2
} // namespace mlwq