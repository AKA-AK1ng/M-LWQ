# Dither / 抖动空间取值说明（Ref / AVX2 / AVX512）

本文总结 M-LWQ 中“抖动（dither）”与“噪声（CBD）”在三个实现版本中的取值空间与采样方式。

## 1) 噪声（s, r）

- 三个版本都使用 `MLWQ_ETA1 = 3` 的 CBD3。输出系数形如 `a-b`，其中 `a,b ∈ [0,3]`（实现中是按 3-bit 计数再做中心化），因此噪声是中心分布，范围约在 `[-3,3]`。
- Ref: `cbd3()` 与 `poly_getnoise_eta1()`。
- AVX2: `avx_cbd3()` / `avx_poly_cbd_eta1()` / `avx_polyvec_getnoise_eta1()`。
- AVX512: `avx512_poly_cbd_eta1()` / `avx512_polyvec_getnoise_eta1()`（内部复用 AVX2 风格 CBD3）。

## 2) 抖动 d_pk / d_u / d_v 的“空间”

- 抖动不是 CBD，而是 **uniform mod** 采样。
- 在 keygen：
  - `d_pk` 通过 `derive_seed_d(seed_A)` 后再加域分离字节 `0xFF` 扩展，模数是 `MLWQ_Q`。
- 在 encrypt：
  - `d_u` 使用 `seed_ct || MLWQ_K`，模数 `MLWQ_Q`。
  - `d_v` 使用 `seed_ct || (MLWQ_K+1)`，模数 `MLWQ_Q`。

## 3) 各版本具体采样实现

### Ref

- `ref_xof_expand_poly_vec()` / `ref_xof_expand_poly()` 从 SHAKE128 字节流取 16-bit 值并做 `% modulus`。
- 因此 dither 系数空间是：`{0,1,...,modulus-1}`。

### AVX2

- `avx_xof_expand_poly_vec()` / `avx_xof_expand_poly()` 同样从 16-bit 值生成，使用 `reduce_u16()`：
  - 若 `modulus` 是 2 的幂，直接按位与；
  - 否则走快速模约减。
- 结果空间仍是 `0..modulus-1`。

### AVX512

- `avx512_xof_expand_poly_vec()` / `avx512_xof_expand_poly()` 与 AVX2 同逻辑（8-lane SHAKE 并行），也使用 `reduce_u16()`。
- `MLWQ_K < 8` 时会回退到 AVX2 的向量扩展实现。
- 结果空间同样是 `0..modulus-1`。

## 4) 以默认参数（NIST L1）直观查看

- `MLWQ_Q = 3329`。
- 因此 `d_pk`、`d_u`、`d_v` 都在同一空间：`[0, 3328]`。

> 现在代码中 dither 统一使用 `MLWQ_Q` 作为 modulus。
