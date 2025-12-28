#ifndef ALIGN_H
#define ALIGN_H

// 只有 C 编译器能看到这些，汇编器跳过
#ifndef __ASSEMBLER__
#include <immintrin.h>
#include <stdint.h>

#define ALIGNED_UINT8(N)        union { uint8_t coeffs[N]; __m256i vec[(N)/32]; }
#define ALIGNED_INT16(N)        union { int16_t coeffs[N]; __m256i vec[(N)/16]; }
#endif

#endif