#include <stdint.h>
#include <immintrin.h>
#include "poly.h"
#include "../common/params.h"

// CBD2: 2 bits for a, 2 bits for b. Val = (a0+a1) - (b0+b1)
// 1 Byte = 2 Coefficients
// Input: 128 bytes -> 256 coeffs
static void cbd2_avx(poly *r, const uint8_t buf[128]) {
    uint32_t t, d;
    int16_t a, b;

    // 为了极致性能，可以使用 AVX2 指令，但其实简单的 64位 SWAR 已经足够快了
    // 因为瓶颈主要在 SHAKE，而不是这里的位操作
    
    for(int i=0; i<MLWQ_N/8; i++) {
        // 每次处理 4 个字节 (8 个系数)
        t = ((uint32_t*)buf)[i];
        
        // 经典的位操作技巧，避免分支
        // d 包含 8 个 4-bit 的块
        // bits 0,1 -> a0, bits 2,3 -> b0 ...
        
        d = t & 0x55555555;
        d += (t >> 1) & 0x55555555;
        
        for(int j=0; j<8; j++) {
            // 从 d 中提取 a 和 b
            // 每次取 2 bits
            a = (d >> (4*j)) & 0x3;     // sum of first 2 bits
            b = (d >> (4*j+2)) & 0x3;   // sum of next 2 bits
            r->coeffs[8*i + j] = a - b;
        }
    }
}

// 供外部调用的包装器
void avx_poly_cbd2(poly *r, const uint8_t buf[128]) {
    cbd2_avx(r, buf);
}