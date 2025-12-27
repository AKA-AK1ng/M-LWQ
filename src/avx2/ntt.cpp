#include "ntt.hpp"
#include <immintrin.h>

namespace mlwq {
namespace avx2 {
namespace ntt {

    using i16 = int16_t;
    using i32 = int32_t;

    const __m256i AVX_Q = _mm256_set1_epi16(3329);
    const __m256i AVX_QINV = _mm256_set1_epi16(-3327);
    const __m256i AVX_R2 = _mm256_set1_epi16(1353);
    const __m256i AVX_F = _mm256_set1_epi16(1441);
    const __m256i AVX_BARRETT_MUL = _mm256_set1_epi16(20159); 

    const i16 zetas[128] = {
      -1044,  -758,  -359, -1517,  1493,  1422,   287,   202,
       -171,   622,  1577,   182,   962, -1202, -1474,  1468,
        573, -1325,   264,   383,  -829,  1458, -1602,  -130,
       -681,  1017,   732,   608, -1542,   411,  -205, -1571,
       1223,   652,  -552,  1015, -1293,  1491,  -282, -1544,
        516,    -8,  -320,  -666, -1618, -1162,   126,  1469,
       -853,   -90,  -271,   830,   107, -1421,  -247,  -951,
       -398,   961, -1508,  -725,   448, -1065,   677, -1275,
      -1103,   430,   555,   843, -1251,   871,  1550,   105,
        422,   587,   177,  -235,  -291,  -460,  1574,  1653,
       -246,   778,  1159,  -147,  -777,  1483,  -602,  1119,
      -1590,   644,  -872,   349,   418,   329,  -156,   -75,
        817,  1097,   603,   610,  1322, -1285, -1465,   384,
      -1215,  -136,  1218, -1335,  -874,   220, -1187, -1659,
      -1185, -1530, -1278,   794, -1510,  -854,  -870,   478,
       -108,  -308,   996,   991,   958, -1460,  1522,  1628
    };

    // --- Scalar Helpers (Needed for fallback) ---
    inline i16 montgomery_reduce(i32 a) {
        i16 u = (i16)(a * -3327);
        i32 t = (i32)((int64_t)u * 3329);
        t = a - t;
        t >>= 16;
        return (i16)t;
    }
    
    inline i16 fqmul(i16 a, i16 b) { return montgomery_reduce((i32)a * b); }

    // [Fix] 添加缺失的 barrett_reduce 定义
    inline i16 barrett_reduce(i16 a) {
        i16 v = ((i32)a * 20159) >> 26;
        v = a - v * 3329;
        return v;
    }

    // --- AVX Helpers ---
    inline __m256i montgomery_mul_avx(__m256i a, __m256i b) {
        __m256i t_low = _mm256_mullo_epi16(a, b);
        __m256i k = _mm256_mullo_epi16(t_low, AVX_QINV);
        __m256i m = _mm256_mulhi_epi16(k, AVX_Q);
        __m256i t_high = _mm256_mulhi_epi16(a, b);
        return _mm256_sub_epi16(t_high, m);
    }

    inline __m256i barrett_reduce_avx(__m256i a) {
        __m256i v = _mm256_mulhi_epi16(a, AVX_BARRETT_MUL);
        v = _mm256_srai_epi16(v, 10); 
        __m256i vq = _mm256_mullo_epi16(v, AVX_Q);
        return _mm256_sub_epi16(a, vq);
    }

    inline void butterfly_avx(__m256i& a, __m256i& b, __m256i zeta) {
        __m256i t = montgomery_mul_avx(b, zeta);
        b = _mm256_sub_epi16(a, t);
        a = _mm256_add_epi16(a, t);
    }

    void ntt_avx(i16* r) {
        __m256i zeta = _mm256_set1_epi16(zetas[1]);
        for(int i=0; i<128; i+=16) {
            __m256i a = _mm256_loadu_si256((__m256i*)&r[i]);
            __m256i b = _mm256_loadu_si256((__m256i*)&r[i+128]);
            butterfly_avx(a, b, zeta);
            _mm256_storeu_si256((__m256i*)&r[i], a);
            _mm256_storeu_si256((__m256i*)&r[i+128], b);
        }

        int k = 2;
        for(int len=64; len>=16; len/=2) {
             for(int start=0; start<256; start+=2*len) {
                 zeta = _mm256_set1_epi16(zetas[k++]);
                 for(int i=start; i<start+len; i+=16) {
                    __m256i a = _mm256_loadu_si256((__m256i*)&r[i]);
                    __m256i b = _mm256_loadu_si256((__m256i*)&r[i+len]);
                    butterfly_avx(a, b, zeta);
                    _mm256_storeu_si256((__m256i*)&r[i], a);
                    _mm256_storeu_si256((__m256i*)&r[i+len], b);
                 }
             }
        }

        // Scalar Fallback
        for(int len = 8; len >= 2; len >>= 1) {
            for(int start = 0; start < 256; start += 2*len) {
                i16 z = zetas[k++];
                for(int j = start; j < start + len; j++) {
                    i16 t = fqmul(z, r[j + len]);
                    r[j + len] = r[j] - t;
                    r[j] = r[j] + t;
                }
            }
        }
    }

    inline void inv_butterfly_avx(__m256i& a, __m256i& b, __m256i zeta) {
        __m256i sum = _mm256_add_epi16(a, b);
        __m256i diff = _mm256_sub_epi16(b, a); // b - a
        a = barrett_reduce_avx(sum);           
        b = montgomery_mul_avx(diff, zeta);
    }

    void invntt_avx(i16* r) {
        int k = 127;
        // Scalar Fallback
        for(int len = 2; len <= 4; len <<= 1) {
            for(int start = 0; start < 256; start += 2*len) {
                i16 z = zetas[k--];
                for(int j = start; j < start + len; j++) {
                    i16 t = r[j];
                    // [Fix] 现在这里能找到 barrett_reduce 了
                    r[j] = barrett_reduce(t + r[j + len]);
                    r[j + len] = r[j + len] - t; 
                    r[j + len] = fqmul(z, r[j + len]);
                }
            }
        }

        // AVX Layer 8
        for(int start = 0; start < 256; start += 16) {
             __m256i zeta = _mm256_set1_epi16(zetas[k--]);
             __m256i v = _mm256_loadu_si256((__m256i*)&r[start]);
             __m256i b_swapped = _mm256_permute2x128_si256(v, v, 0x01); 
             __m256i a = v; 
             __m256i sum = _mm256_add_epi16(a, b_swapped);
             __m256i diff = _mm256_sub_epi16(b_swapped, a); 
             a = barrett_reduce_avx(sum);          
             b_swapped = montgomery_mul_avx(diff, zeta); 
             __m256i final_v = _mm256_inserti128_si256(a, _mm256_castsi256_si128(b_swapped), 1);
             _mm256_storeu_si256((__m256i*)&r[start], final_v);
        }

        // AVX Large Layers
        for(int len = 16; len <= 128; len <<= 1) {
            for(int start = 0; start < 256; start += 2*len) {
                __m256i zeta = _mm256_set1_epi16(zetas[k--]);
                for(int j = start; j < start + len; j+=16) {
                    __m256i a = _mm256_loadu_si256((__m256i*)&r[j]);
                    __m256i b = _mm256_loadu_si256((__m256i*)&r[j+len]);
                    inv_butterfly_avx(a, b, zeta);
                    _mm256_storeu_si256((__m256i*)&r[j], a);
                    _mm256_storeu_si256((__m256i*)&r[j+len], b);
                }
            }
        }
        
        // Final Factor Scaling
        const __m256i v_f = AVX_F;
        for(int i=0; i<256; i+=16) {
             __m256i a = _mm256_loadu_si256((__m256i*)&r[i]);
             a = montgomery_mul_avx(a, v_f);
             _mm256_storeu_si256((__m256i*)&r[i], a);
        }
    }

    inline __m256i swap_adjacent_pairs(__m256i v) {
        v = _mm256_shufflelo_epi16(v, 0xB1);
        v = _mm256_shufflehi_epi16(v, 0xB1);
        return v;
    }

    void poly_basemul_avx(i16* r, const i16* a, const i16* b) {
        for(int i=0; i<256/16; i++) { 
            i16 z0 = zetas[64 + i*4 + 0];
            i16 z1 = zetas[64 + i*4 + 1];
            i16 z2 = zetas[64 + i*4 + 2];
            i16 z3 = zetas[64 + i*4 + 3];
            __m256i vzeta = _mm256_set_epi16(-z3, -z3, z3, z3, -z2, -z2, z2, z2, -z1, -z1, z1, z1, -z0, -z0, z0, z0);
            __m256i va = _mm256_loadu_si256((__m256i*)&a[i*16]);
            __m256i vb = _mm256_loadu_si256((__m256i*)&b[i*16]);
            __m256i va_swap = swap_adjacent_pairs(va);
            __m256i v_prod_rot = montgomery_mul_avx(va_swap, vb); 
            __m256i r1_vec = _mm256_add_epi16(v_prod_rot, swap_adjacent_pairs(v_prod_rot));
            __m256i v_prod = montgomery_mul_avx(va, vb);
            __m256i v_prod_swapped = swap_adjacent_pairs(v_prod);
            __m256i v_term_zeta = montgomery_mul_avx(v_prod_swapped, vzeta);
            __m256i r0_vec = _mm256_add_epi16(v_prod, v_term_zeta);
            __m256i res = _mm256_blend_epi16(r0_vec, r1_vec, 0xAA);
            _mm256_storeu_si256((__m256i*)&r[i*16], res);
        }
    }

    void poly_tomont_avx(i16* r) {
        for(int i=0; i<256; i+=16) {
            __m256i a = _mm256_loadu_si256((__m256i*)&r[i]);
            a = montgomery_mul_avx(a, AVX_R2);
            _mm256_storeu_si256((__m256i*)&r[i], a);
        }
    }

    // AVX2 Implementation of poly_mul_ntt
    std::vector<i16> poly_mul_ntt(const std::vector<i16>& a, const std::vector<i16>& b) {
        std::vector<i16> res(256);
        std::vector<i16> ta = a;
        std::vector<i16> tb = b;

        poly_tomont_avx(ta.data());
        poly_tomont_avx(tb.data());
        ntt_avx(ta.data());
        ntt_avx(tb.data());
        poly_basemul_avx(res.data(), ta.data(), tb.data());
        invntt_avx(res.data());
        
        // Final Normalize
        const __m256i v_one = _mm256_set1_epi16(1); 
        for(int i=0; i<256; i+=16) {
            __m256i val = _mm256_loadu_si256((__m256i*)&res[i]);
            val = montgomery_mul_avx(val, v_one); 
            val = montgomery_mul_avx(val, v_one); 
            __m256i mask = _mm256_cmpgt_epi16(_mm256_setzero_si256(), val); 
            __m256i add_q = _mm256_and_si256(mask, AVX_Q);
            val = _mm256_add_epi16(val, add_q);
            _mm256_storeu_si256((__m256i*)&res[i], val);
        }
        return res;
    }

}
}
}