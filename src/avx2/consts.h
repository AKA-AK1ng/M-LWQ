#ifndef AVX_CONSTS_H
#define AVX_CONSTS_H

#include "params.h"

// --------------------------------------------------------
// 第一部分：偏移量宏 (C 和 汇编 都需要，保持可见)
// --------------------------------------------------------
#define _16XQ            0
#define _16XQINV        16
#define _16XV           32
#define _16XFLO         48
#define _16XFHI         64
#define _16XMONTSQLO    80
#define _16XMONTSQHI    96
#define _16XMASK       112
#define _REVIDXB       128
#define _REVIDXD       144
#define _ZETAS_EXP     160
#define	_16XSHIFT      624

// --------------------------------------------------------
// 第二部分：汇编专用宏 (仅汇编可见)
// --------------------------------------------------------
#ifdef __ASSEMBLER__
// 处理不同操作系统的函数名修饰 (macOS/Win 需要下划线前缀)
#if defined(__WIN32__) || defined(__APPLE__)
#define decorate(s) _##s
#define cdecl2(s) decorate(s)
#define cdecl(s) cdecl2(KYBER_NAMESPACE(##s))
#else
#define cdecl(s) KYBER_NAMESPACE(##s)
#endif
#endif

// --------------------------------------------------------
// 第三部分：C 语言声明 (仅 C 可见，汇编器必须跳过！)
// --------------------------------------------------------
#ifndef __ASSEMBLER__
#include "align.h"

typedef ALIGNED_INT16(640) qdata_t;
#define qdata KYBER_NAMESPACE(qdata)
extern const qdata_t qdata;
#endif

#endif