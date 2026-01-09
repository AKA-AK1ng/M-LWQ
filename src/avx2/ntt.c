#include "ntt.h"
#include "consts.h"
#include <immintrin.h> // 需要 __m256i 定义

extern void mlwq_avx_ntt_avx(__m256i *r, const __m256i *qdata);
extern void mlwq_avx_invntt_avx(__m256i *r, const __m256i *qdata);
extern void mlwq_avx_basemul_avx(__m256i *r, const __m256i *a, const __m256i *b, const __m256i *qdata);
extern void mlwq_avx_reduce_avx(__m256i *r, const __m256i *qdata);

void avx_ntt(int16_t *r) {
    mlwq_avx_ntt_avx((__m256i*)r, qdata.vec);
}

void avx_invntt(int16_t *r) {
    mlwq_avx_invntt_avx((__m256i*)r, qdata.vec);
}

void avx_basemul(int16_t *r, const int16_t *a, const int16_t *b) {
    mlwq_avx_basemul_avx((__m256i*)r, (const __m256i*)a, (const __m256i*)b, qdata.vec);
}

void avx_reduce(int16_t *r) {
    mlwq_avx_reduce_avx((__m256i*)r, qdata.vec);
}

void avx_poly_mul_ntt(poly *res, const poly *a, const poly *b) {
    // [优化] 零拷贝模式
    // 因为 poly 结构体已经 32 字节对齐，我们可以直接转换指针
    
    // 注意：Kyber 的 NTT 是 In-Place (原地) 的，会破坏输入数据
    // 但这里的函数签名 a 和 b 是 const，我们不能修改它们
    // 所以我们仍然需要拷贝 *一次* 到 res，或者使用临时变量
    // 但我们可以省去 "拷贝到 buffer -> 计算 -> 拷贝回 res" 的过程
    
    // 方案：直接在 res 上进行计算
    // 1. 拷贝 a 到 res (这也是必须的，因为 NTT 是原地的)
    *res = *a; 
    
    // 2. 我们还需要一个临时变量放 b，因为不能修改 const b
    poly tmp_b = *b; 

    // 3. 直接调用汇编，无需中间对齐 buffer
    mlwq_avx_ntt_avx((__m256i*)res->coeffs, qdata.vec);
    mlwq_avx_ntt_avx((__m256i*)tmp_b.coeffs, qdata.vec);

    // 4. Pointwise Mul (结果存入 res)
    mlwq_avx_basemul_avx((__m256i*)res->coeffs, 
                         (__m256i*)res->coeffs, 
                         (__m256i*)tmp_b.coeffs, 
                         qdata.vec);

    // 5. Inverse NTT
    mlwq_avx_invntt_avx((__m256i*)res->coeffs, qdata.vec);

    // 6. 最终规约 (AVX2 批量规约)
    avx_reduce(res->coeffs);
}
