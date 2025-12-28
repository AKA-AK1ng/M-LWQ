#ifndef PARAMS_H
#define PARAMS_H

// === 配置 NIST 安全等级 (1, 3, 5) ===
#ifndef NIST_LEVEL
#define NIST_LEVEL 1
#endif

// [新增] 补充这两个定义
// 12-bit 打包: 256 * 12 / 8 = 384 字节
#define MLWQ_POLYBYTES 384

// 整个向量的打包大小 = K * 单个多项式大小
#define MLWQ_POLYVECBYTES (MLWQ_K * MLWQ_POLYBYTES)

// 2. [修复] 压缩打包 (密文部分)
// PolyVec u (10-bit): 256 * 10 / 8 = 320 bytes
#define MLWQ_POLYVECCOMPRESSEDBYTES (MLWQ_K * 320)

// Poly v (4-bit): 256 * 4 / 8 = 128 bytes
#define MLWQ_POLYCOMPRESSEDBYTES 128

#define MLWQ_N 256
#define MLWQ_Q 3329
#define MLWQ_ETA 2
#define MLWQ_ETA1 2

#if NIST_LEVEL == 1
    #define PARAM_NAME "M-LWQ-512 (C-Imp L1)"
    #define MLWQ_K 2
    #define P_PK 512   // 1<<9
    #define P_U  1024   // 1<<9
    #define P_V  32    // 1<<5
#elif NIST_LEVEL == 3
    #define PARAM_NAME "M-LWQ-768 (C-Imp L3)"
    #define MLWQ_K 3
    #define P_PK 1024  // 1<<10
    #define P_U  1024
    #define P_V  32
#elif NIST_LEVEL == 5
    #define PARAM_NAME "M-LWQ-1024 (C-Imp L5)"
    #define MLWQ_K 4
    #define P_PK 2048  // 1<<11
    #define P_U  2048
    #define P_V  64
#else
    #error "Invalid NIST_LEVEL"
#endif

// 种子长度
#define SEEDBYTES 32
#define HASHBYTES 32


#endif