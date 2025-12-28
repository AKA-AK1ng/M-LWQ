#include "poly.h"
#include "ntt.h"
#include "reduce.h"
#include "../common/params.h"


static inline int16_t load24_littleendian(const uint8_t x[3]) {
  return (int16_t)((uint16_t)x[0] | ((uint16_t)x[1] << 8) | ((uint16_t)x[2] << 16));
}

static void cbd3(poly *r, const uint8_t buf[3*MLWQ_N/4]) {
  unsigned int i, j;
  uint32_t t, d;
  int16_t a, b;

  // 每次处理 3 字节 -> 生成 4 个系数
  for(i=0; i<MLWQ_N/4; i++) {
    t = load24_littleendian(buf+3*i);
    d = t & 0x00249249;
    d += (t>>1) & 0x00249249;
    d += (t>>2) & 0x00249249;

    for(j=0; j<4; j++) {
      a = (d >> (6*j+0)) & 0x7;
      b = (d >> (6*j+3)) & 0x7;
      r->coeffs[4*i+j] = a - b;
    }
  }
}

// -------------------------------------------------------------------------
// [新增] 多项式采样接口 (PRF + CBD)
// -------------------------------------------------------------------------
void ref_poly_getnoise_eta1(poly *r, const uint8_t *seed, uint8_t nonce) {
    uint8_t buf[MLWQ_ETA1*MLWQ_N/4]; // Eta=3 -> 192 bytes
    uint8_t extseed[33]; // seed + nonce
    
    // 1. 准备种子: seed || nonce
    for(int i=0; i<32; i++) extseed[i] = seed[i];
    extseed[32] = nonce;
    
    // 2. PRF (使用 SHAKE256 生成固定长度字节)
    shake256(buf, sizeof(buf), extseed, 33);
    
    // 3. CBD 解析
    cbd3(r, buf);
}

// -------------------------------------------------------------------------
// 1. NTT 适配层
// -------------------------------------------------------------------------
static void poly_ntt(poly *r) { ntt(r->coeffs); }
static void poly_invntt(poly *r) { invntt(r->coeffs); }

static void poly_basemul(poly *r, const poly *a, const poly *b) {
    for(int i = 0; i < MLWQ_N / 4; i++) {
        basemul(&r->coeffs[4 * i], &a->coeffs[4 * i], &b->coeffs[4 * i], zetas[64 + i]);
        basemul(&r->coeffs[4 * i + 2], &a->coeffs[4 * i + 2], &b->coeffs[4 * i + 2], -zetas[64 + i]);
    }
}

// -------------------------------------------------------------------------
// 2. [核心优化] 极速打包与解包 (Packing/Unpacking)
// -------------------------------------------------------------------------
// 消除 KEM KeyGen 与 PKE KeyGen 差距的关键！

// 将多项式序列化为字节流 (12-bit -> 8-bit)
// 优化策略：一次处理 2 个系数 -> 生成 3 个字节
void ref_poly_tobytes(uint8_t r[MLWQ_POLYBYTES], const poly *a)
{
  int i;
  uint16_t t0, t1, t2, t3, t4, t5, t6, t7;

  // 每次循环处理 8 个系数 (4 组) -> 生成 12 个字节
  // 256 / 8 = 32 次循环
  for(i=0; i<MLWQ_N/8; i++)
  {
    // Load 8 coeffs
    t0 = a->coeffs[8*i+0]; t0 += ((int16_t)t0 >> 15) & MLWQ_Q;
    t1 = a->coeffs[8*i+1]; t1 += ((int16_t)t1 >> 15) & MLWQ_Q;
    t2 = a->coeffs[8*i+2]; t2 += ((int16_t)t2 >> 15) & MLWQ_Q;
    t3 = a->coeffs[8*i+3]; t3 += ((int16_t)t3 >> 15) & MLWQ_Q;
    t4 = a->coeffs[8*i+4]; t4 += ((int16_t)t4 >> 15) & MLWQ_Q;
    t5 = a->coeffs[8*i+5]; t5 += ((int16_t)t5 >> 15) & MLWQ_Q;
    t6 = a->coeffs[8*i+6]; t6 += ((int16_t)t6 >> 15) & MLWQ_Q;
    t7 = a->coeffs[8*i+7]; t7 += ((int16_t)t7 >> 15) & MLWQ_Q;

    // Pack 1 (t0, t1)
    r[12*i+0] = (uint8_t)(t0 >> 0);
    r[12*i+1] = (uint8_t)((t0 >> 8) | (t1 << 4));
    r[12*i+2] = (uint8_t)(t1 >> 4);

    // Pack 2 (t2, t3)
    r[12*i+3] = (uint8_t)(t2 >> 0);
    r[12*i+4] = (uint8_t)((t2 >> 8) | (t3 << 4));
    r[12*i+5] = (uint8_t)(t3 >> 4);

    // Pack 3 (t4, t5)
    r[12*i+6] = (uint8_t)(t4 >> 0);
    r[12*i+7] = (uint8_t)((t4 >> 8) | (t5 << 4));
    r[12*i+8] = (uint8_t)(t5 >> 4);

    // Pack 4 (t6, t7)
    r[12*i+9] = (uint8_t)(t6 >> 0);
    r[12*i+10] = (uint8_t)((t6 >> 8) | (t7 << 4));
    r[12*i+11] = (uint8_t)(t7 >> 4);
  }
}

// -------------------------------------------------------------------------
// [极致优化] 解包: 8路循环展开
// -------------------------------------------------------------------------
void ref_poly_frombytes(poly *r, const uint8_t a[MLWQ_POLYBYTES])
{
  int i;
  for(i=0; i<MLWQ_N/8; i++)
  {
    // Unpack 1
    r->coeffs[8*i+0] = ((a[12*i+0] >> 0) | ((uint16_t)a[12*i+1] << 8)) & 0xFFF;
    r->coeffs[8*i+1] = ((a[12*i+1] >> 4) | ((uint16_t)a[12*i+2] << 4)) & 0xFFF;

    // Unpack 2
    r->coeffs[8*i+2] = ((a[12*i+3] >> 0) | ((uint16_t)a[12*i+4] << 8)) & 0xFFF;
    r->coeffs[8*i+3] = ((a[12*i+4] >> 4) | ((uint16_t)a[12*i+5] << 4)) & 0xFFF;

    // Unpack 3
    r->coeffs[8*i+4] = ((a[12*i+6] >> 0) | ((uint16_t)a[12*i+7] << 8)) & 0xFFF;
    r->coeffs[8*i+5] = ((a[12*i+7] >> 4) | ((uint16_t)a[12*i+8] << 4)) & 0xFFF;

    // Unpack 4
    r->coeffs[8*i+6] = ((a[12*i+9] >> 0) | ((uint16_t)a[12*i+10] << 8)) & 0xFFF;
    r->coeffs[8*i+7] = ((a[12*i+10] >> 4) | ((uint16_t)a[12*i+11] << 4)) & 0xFFF;
  }
}

// -------------------------------------------------------------------------
// 3. [核心优化] 快速压缩与解压缩 (Compress/Decompress)
// -------------------------------------------------------------------------
// 用于 Encaps/Decaps 中的密文处理

// 快速除法常数 (针对 Q=3329)
// 1/3329 approx 322376 / 2^30
static inline int32_t fast_div_3329(int32_t val) {
    return (int32_t)(((int64_t)val * 322376) >> 30);
}

// 量化 (带抖动)
void ref_poly_quantize(poly *res, const poly *v, const poly *d, int32_t P) {
    int32_t mask = P - 1;
    for(int i = 0; i < MLWQ_N; ++i) {
        int32_t val = (int32_t)v->coeffs[i] + d->coeffs[i];
        int32_t temp = val * P;
        int32_t floor = fast_div_3329(temp);
        res->coeffs[i] = floor & mask;
    }
}

// 反量化 (恢复)
void ref_poly_dequantize(poly *res, const poly *b, int32_t P) {
    for(int i = 0; i < MLWQ_N; ++i) {
        // v = (b * Q + Q/2) / P
        // 尽量使用乘法，但这里为了精度保持除法 (非性能热点)
        int32_t v = (int32_t)b->coeffs[i] * MLWQ_Q + (MLWQ_Q / 2);
        res->coeffs[i] = v / P;
    }
}

// -------------------------------------------------------------------------
// 4. 基础算术 & 矩阵乘法 (保持之前的 NTT 优化)
// -------------------------------------------------------------------------

void ref_poly_add(poly *res, const poly *a, const poly *b) {
    for(int i = 0; i < MLWQ_N; i+=8) {
        res->coeffs[i+0] = barrett_reduce(a->coeffs[i+0] + b->coeffs[i+0]);
        res->coeffs[i+1] = barrett_reduce(a->coeffs[i+1] + b->coeffs[i+1]);
        res->coeffs[i+2] = barrett_reduce(a->coeffs[i+2] + b->coeffs[i+2]);
        res->coeffs[i+3] = barrett_reduce(a->coeffs[i+3] + b->coeffs[i+3]);
        res->coeffs[i+4] = barrett_reduce(a->coeffs[i+4] + b->coeffs[i+4]);
        res->coeffs[i+5] = barrett_reduce(a->coeffs[i+5] + b->coeffs[i+5]);
        res->coeffs[i+6] = barrett_reduce(a->coeffs[i+6] + b->coeffs[i+6]);
        res->coeffs[i+7] = barrett_reduce(a->coeffs[i+7] + b->coeffs[i+7]);
    }
}

void ref_poly_sub(poly *res, const poly *a, const poly *b) {
    for(int i = 0; i < MLWQ_N; i+=8) {
        res->coeffs[i+0] = barrett_reduce(a->coeffs[i+0] - b->coeffs[i+0]);
        res->coeffs[i+1] = barrett_reduce(a->coeffs[i+1] - b->coeffs[i+1]);
        res->coeffs[i+2] = barrett_reduce(a->coeffs[i+2] - b->coeffs[i+2]);
        res->coeffs[i+3] = barrett_reduce(a->coeffs[i+3] - b->coeffs[i+3]);
        res->coeffs[i+4] = barrett_reduce(a->coeffs[i+4] - b->coeffs[i+4]);
        res->coeffs[i+5] = barrett_reduce(a->coeffs[i+5] - b->coeffs[i+5]);
        res->coeffs[i+6] = barrett_reduce(a->coeffs[i+6] - b->coeffs[i+6]);
        res->coeffs[i+7] = barrett_reduce(a->coeffs[i+7] - b->coeffs[i+7]);
    }
}

void ref_poly_mul_ntt(poly *res, const poly *a, const poly *b) {
    poly ta = *a;
    poly tb = *b;
    poly_ntt(&ta);
    poly_ntt(&tb);
    poly_basemul(res, &ta, &tb);
    poly_invntt(res);
    for(int i=0; i<MLWQ_N; i++) res->coeffs[i] = barrett_reduce(res->coeffs[i]);
}

void ref_poly_matrix_vec_mul(poly_vec *res, const poly_matrix *A, const poly_vec *s) {
    poly_vec s_ntt = *s;
    for(int i=0; i<MLWQ_K; i++) poly_ntt(&s_ntt.vec[i]);

    for(int i = 0; i < MLWQ_K; ++i) {
        poly acc;
        for(int k=0; k<MLWQ_N; k++) acc.coeffs[k] = 0;
        for(int j = 0; j < MLWQ_K; ++j) {
            poly t_a = A->row[i].vec[j];
            poly t_prod;
            poly_ntt(&t_a);
            poly_basemul(&t_prod, &t_a, &s_ntt.vec[j]);
            for(int k=0; k<MLWQ_N; k++) acc.coeffs[k] += t_prod.coeffs[k];
        }
        poly_invntt(&acc);
        for(int k=0; k<MLWQ_N; k++) res->vec[i].coeffs[k] = barrett_reduce(acc.coeffs[k]);
    }
}

void ref_poly_vec_transpose_mul(poly *res, const poly_vec *a_t, const poly_vec *b) {
    poly_vec a_ntt = *a_t;
    poly_vec b_ntt = *b;
    for(int i=0; i<MLWQ_K; i++) {
        poly_ntt(&a_ntt.vec[i]);
        poly_ntt(&b_ntt.vec[i]);
    }
    
    poly acc;
    for(int k=0; k<MLWQ_N; k++) acc.coeffs[k] = 0;
    
    for(int i = 0; i < MLWQ_K; ++i) {
        poly t_prod;
        poly_basemul(&t_prod, &a_ntt.vec[i], &b_ntt.vec[i]);
        for(int k=0; k<MLWQ_N; k++) acc.coeffs[k] += t_prod.coeffs[k];
    }
    poly_invntt(&acc);
    for(int k=0; k<MLWQ_N; k++) res->coeffs[k] = barrett_reduce(acc.coeffs[k]);
}

// -------------------------------------------------------------------------
// 5. 消息编码/解码
// -------------------------------------------------------------------------

void ref_poly_msg_encode(poly *res, const uint8_t *msg) {
    int32_t scale = MLWQ_Q / 2;
    for(int i = 0; i < MLWQ_N; ++i) {
        int bit = (msg[i/8] >> (i%8)) & 1;
        res->coeffs[i] = bit ? scale : 0;
    }
}

void ref_poly_msg_decode(uint8_t *msg, const poly *p) {
    for(int i=0; i<32; i++) msg[i] = 0; 
    int32_t lower = MLWQ_Q / 4;
    int32_t upper = 3 * MLWQ_Q / 4;
    for(int i = 0; i < MLWQ_N; ++i) {
        int16_t val = p->coeffs[i];
        int32_t t = (int32_t)val;
        if (t < 0) t += MLWQ_Q;
        int bit = (t > lower && t < upper) ? 1 : 0;
        if(bit) msg[i/8] |= (1 << (i%8));
    }
}

int ref_check_poly_eq(const poly *a, const poly *b) {
    for(int i = 0; i < MLWQ_N; ++i)
        if (a->coeffs[i] != b->coeffs[i]) return 0;
    return 1;
}

