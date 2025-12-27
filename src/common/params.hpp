#pragma once
#include <cstdint>

// ==========================================
// [配置] 选择安全等级 (1, 3, or 5)
// ==========================================
#ifndef NIST_LEVEL
#define NIST_LEVEL 1 // 修改此处切换等级: 1, 3, 5
#endif

namespace mlwq {
namespace params {

    enum QuantizationType { QUANT_SCALAR };
    constexpr QuantizationType Q_MODE = QUANT_SCALAR;

    // 通用参数 (所有等级共享)
    constexpr int32_t N = 256;
    constexpr int32_t Q = 3329;
    constexpr int32_t ETA = 2;
    constexpr int32_t MSG_MODULUS = 2; // 消息模数 (1bit per coeff)

    // --- 等级特定参数 ---
#if NIST_LEVEL == 1
    constexpr char PARAM_SET_NAME[] = "M-LWQ-512 (NIST Level 1)";
    constexpr int32_t K = 2;             // 矩阵维度 2x2
    constexpr int32_t D_PK_BITS = 9;     // 公钥量化比特
    constexpr int32_t D_U_BITS = 9;      // 密文 u 量化比特
    constexpr int32_t D_V_BITS = 5;      // 密文 v 量化比特

#elif NIST_LEVEL == 3
    constexpr char PARAM_SET_NAME[] = "M-LWQ-768 (NIST Level 3)";
    constexpr int32_t K = 3;             // 矩阵维度 3x3
    constexpr int32_t D_PK_BITS = 10;    // 通常 L3 需要更高的精度
    constexpr int32_t D_U_BITS = 10;
    constexpr int32_t D_V_BITS = 5;

#elif NIST_LEVEL == 5
    constexpr char PARAM_SET_NAME[] = "M-LWQ-1024 (NIST Level 5)";
    constexpr int32_t K = 4;             // 矩阵维度 4x4
    constexpr int32_t D_PK_BITS = 11;
    constexpr int32_t D_U_BITS = 11;
    constexpr int32_t D_V_BITS = 6;

#else
    #error "Invalid NIST_LEVEL. Please set to 1, 3, or 5."
#endif

    // 自动派生参数
    constexpr int32_t P_PK = (1 << D_PK_BITS);
    constexpr int32_t P_U = (1 << D_U_BITS);
    constexpr int32_t P_V = (1 << D_V_BITS);
}
}