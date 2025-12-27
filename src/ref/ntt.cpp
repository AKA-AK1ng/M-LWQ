#include "ntt.hpp"

namespace mlwq {
namespace ref {
namespace ntt {

    using i16 = int16_t;
    using i32 = int32_t;

    const i16 Q = 3329;
    const i16 QINV = -3327; 
    const i16 R2 = 1353;    
    const i16 F_FACTOR = 1441; 

    // [Verified] Exactly 128 elements (16 rows * 8 cols)
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

    inline i16 montgomery_reduce(i32 a) {
        i16 u = (i16)(a * (i32)QINV);
        i32 t = (i32)((int64_t)u * Q);
        t = a - t;
        t >>= 16;
        return (i16)t;
    }
    
    inline i16 fqmul(i16 a, i16 b) {
        return montgomery_reduce((i32)a * b);
    }

    inline i16 barrett_reduce(i16 a) {
        i16 v = ((i32)a * 20159) >> 26;
        v = a - v * 3329;
        return v;
    }

    void ntt_scalar(i16* r) {
        unsigned int len, start, j, k=1;
        i16 t, zeta;
        for(len = 128; len >= 2; len >>= 1) {
            for(start = 0; start < 256; start = j + len) {
                zeta = zetas[k++];
                for(j = start; j < start + len; j++) {
                    t = fqmul(zeta, r[j + len]);
                    r[j + len] = r[j] - t;
                    r[j] = r[j] + t;
                }
            }
        }
    }
    
    void invntt_scalar(i16* r) {
        unsigned int start, len, j, k=127;
        i16 t, zeta;
        for(len = 2; len <= 128; len <<= 1) {
            for(start = 0; start < 256; start = j + len) {
                zeta = zetas[k--];
                for(j = start; j < start + len; j++) {
                    t = r[j];
                    r[j] = barrett_reduce(t + r[j + len]);
                    r[j + len] = r[j + len] - t; 
                    r[j + len] = fqmul(zeta, r[j + len]);
                }
            }
        }
        for(j = 0; j < 256; j++) r[j] = fqmul(r[j], F_FACTOR); 
    }

    void basemul_scalar(std::vector<i16>& r, const std::vector<i16>& a, const std::vector<i16>& b) {
        for (int i = 0; i < 256 / 4; i++) {
            i16 zeta = zetas[64 + i];
            
            i16 t1 = fqmul(a[4*i+1], b[4*i+1]); t1 = fqmul(t1, zeta);
            i16 t0 = fqmul(a[4*i], b[4*i]); r[4*i] = t1 + t0;

            r[4*i+1] = fqmul(a[4*i], b[4*i+1]) + fqmul(a[4*i+1], b[4*i]);
            
            i16 m_zeta = -zeta;
            i16 t3 = fqmul(a[4*i+3], b[4*i+3]); t3 = fqmul(t3, m_zeta);
            i16 t2 = fqmul(a[4*i+2], b[4*i+2]); r[4*i+2] = t3 + t2;

            r[4*i+3] = fqmul(a[4*i+2], b[4*i+3]) + fqmul(a[4*i+3], b[4*i+2]);
        }
    }

    // Reference implementation (uses scalar helpers)
    std::vector<i16> poly_mul_ntt(const std::vector<i16>& a, const std::vector<i16>& b) {
        std::vector<i16> res(256);
        std::vector<i16> ta = a;
        std::vector<i16> tb = b;

        // 1. To NTT Domain
        for(int i=0; i<256; i++) { ta[i] = fqmul(ta[i], R2); tb[i] = fqmul(tb[i], R2); }
        ntt_scalar(ta.data());
        ntt_scalar(tb.data());

        // 2. Base Mul
        basemul_scalar(res, ta, tb);

        // 3. Inv NTT
        invntt_scalar(res.data());

        // 4. Reduce
        for(int i=0; i<256; i++) {
            i16 val = res[i];
            val = montgomery_reduce((i32)val); 
            val = montgomery_reduce((i32)val); 
            res[i] = (val < 0) ? val + Q : val;
        }
        return res;
    }

}
}
}