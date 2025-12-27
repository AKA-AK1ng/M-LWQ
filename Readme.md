# M-LWQ-PKE: High-Performance C++ Implementation with AVX2 Acceleration

[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://isocpp.org/std/the-standard)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![AVX2](https://img.shields.io/badge/Arch-AVX2-red.svg)](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions)

This repository contains an optimized C++17 implementation of the **M-LWQ (Module-Learning With Quantization)** Public Key Encryption (PKE) scheme.

It serves as a reference implementation and high-precision benchmarking tool, featuring **AVX2 SIMD acceleration** and **Number Theoretic Transform (NTT)** for polynomial arithmetic.

## 1. Core Concept (M-LWQ)

M-LWQ is a novel post-quantum cryptosystem built on the **Learning With Quantization (LWQ)** problem. It replaces the additive Gaussian error sampling (used in Kyber/LWE) with a deterministic **dithered quantization process**, achieving:

1.  **Tight Security:** Security reduction to Module-LWE.
2.  **Extreme Compactness:** Eliminates the need to transmit or store large error terms.

## 2. Implementation Features

This project focuses on **performance** and **correctness verification**.

### 🚀 Optimization Highlights
* **AVX2 Acceleration:**
    * Explicit **AVX2 Intrinsics** for polynomial addition, subtraction, and component-wise operations.
    * Vectorized **Base Multiplication** within the NTT domain.
* **Fast NTT (Number Theoretic Transform):**
    * Replaces the naive $O(N^2)$ multiplication with an efficient $O(N \log N)$ **NTT** implementation compatible with Kyber parameters ($N=256, Q=3329$).
    * Includes **Barrett Reduction** for fast modular arithmetic.
* **Scalar vs. AVX2 Benchmark:**
    * A built-in benchmarking suite that runs the cryptosystem in both **Scalar (Pure C++)** and **AVX2** modes side-by-side to demonstrate speedups.

### 🛠 Algorithms
* **KeyGen / Encrypt / Decrypt:** Complete PKE flow implementation.
* **Quantization:** Efficient implementation of $\mathbb{Z}$ (Scalar) lattice quantization.
* **SHAKE-128:** Self-contained implementation (no external crypto libraries required).

## 3. Build and Run

### Dependencies
* **Compiler:** C++17 compatible (GCC, Clang, or MSVC).
* **Hardware:** CPU with **AVX2** and **FMA** instruction set support (required for the accelerated path).
* **CMake:** Version 3.10 or higher.

### Compilation

The `CMakeLists.txt` is configured to automatically enable `-mavx2`, `-mfma`, and `-O3` optimizations.

```bash
# 1. Clone the repository
git clone [https://github.com/Make1205/M-LWQ.git](https://github.com/Make1205/M-LWQ.git)
cd M-LWQ

# 2. Create build directory
mkdir build
cd build

# 3. Configure and Build
cmake ..
make
./mlwq_demo

```
## 4. Expected Output


The program will run the full PKE suite in Scalar Mode followed by AVX2 Mode, verifying decryption correctness in every round, and finally producing a speedup report.

(Sample output on an Intel Core i7 CPU)

```
=== M-LWQ Comprehensive Performance Report ===
M-LWQ-512 (NIST Level 1)
N=256, K=2

>>> Running: Scalar Mode (1000 rounds)...
   [PASS] Correctness verified.
>>> Running: AVX2 Mode (1000 rounds)...
   [PASS] Correctness verified.

>>> PART 1: Internal Breakdown (Where is time spent?)

--------------------------------------------------------------------------------------
 PKE KeyGen Breakdown (Detailed)
--------------------------------------------------------------------------------------
Sub-Component       Scalar (cyc)   AVX2 (cyc)     Speedup        Scalar %
GenMatrix (A)       29766          14509          2.05x          31.1%
Sample (s)          6186           5095           1.21x          6.5%
GenDither           14751          12786          1.15x          15.4%
Arith (A*s)         44453          33164          1.34x          46.4%
Quantize            577            679            0.85x          0.6%

--------------------------------------------------------------------------------------
 PKE Encrypt Breakdown (Detailed)
--------------------------------------------------------------------------------------
Sub-Component       Scalar (cyc)   AVX2 (cyc)     Speedup        Scalar %
GenMatrix (A)       28984          13953          2.08x          20.9%
Sample (r)          12305          12139          1.01x          8.9%
GenDither           20898          21551          0.97x          15.1%
Arith (u)           51218          32812          1.56x          36.9%
Arith (v)           24705          19186          1.29x          17.8%
Quantize            747            702            1.06x          0.5%

--------------------------------------------------------------------------------------
 PKE Decrypt Breakdown (Detailed)
--------------------------------------------------------------------------------------
Sub-Component       Scalar (cyc)   AVX2 (cyc)     Speedup        Scalar %
DeQuantize          4064           3621           1.12x          16.1%
Arith (v-su)        21011          16578          1.27x          83.3%
Decode              144            127            1.13x          0.6%


>>> PART 2: Core Component Comparison (Quantize vs Sample)
----------------------------------------------------------------------------------------------
Component   Mode        Quantize        Sample          Alg. Efficiency       AVX Improvement     
----------------------------------------------------------------------------------------------
PK / u      Scalar      286             4174            14.57x                1.00x (Ref)         
PK / u      AVX2        256             4130            16.9x                 1.12x               


>>> PART 3: PKE Full Flow Summary (Total Time)
----------------------------------------------------------------------------------------------
Operation           Scalar Cycles     AVX2 Cycles       Speedup
----------------------------------------------------------------------------------------------
PKE KeyGen          94290             64201             1.47x
PKE Encrypt         119502            88485             1.35x
PKE Decrypt         24372             20641             1.18x


>>> PART 4: KEM Full Flow Summary (IND-CCA2)
----------------------------------------------------------------------------------------------
Operation           Scalar Cycles     AVX2 Cycles       Speedup
----------------------------------------------------------------------------------------------
KEM KeyGen          110904            75820             1.46x
KEM Encaps          146720            106138            1.38x
KEM Decaps          159642            124261            1.28x
----------------------------------------------------------------------------------------------

[FINAL] All checks passed! Implementation is correct.
```
Note: Speedup factors depend on your specific CPU architecture. The NTT implementation reduces complexity from quadratic to log-linear, providing significant gains even without AVX, while AVX2 further accelerates the vectorized operations.



## 5. Project Structure
```
.
├── CMakeLists.txt          # CMake config (Auto-enables AVX2)
├── src/
│   ├── main.cpp            # Dual-mode benchmark runner
│   ├── mlwq.hpp/cpp        # Core M-LWQ PKE algorithms
│   ├── poly.hpp/cpp        # Poly arithmetic (Add/Sub AVX2 intrinsics)
│   ├── ntt.hpp/cpp         # NTT implementation & AVX2 BaseMul
│   ├── params.hpp          # Global params & runtime AVX switch
│   ├── random.hpp/cpp      # Random sampling
│   ├── xof.hpp/cpp         # SHAKE-128 wrapper
│   └── cycles.hpp          # RDTSC cycle counter
└── ...
```
<!-- ## 6. Academic Citation
If you use this work in your research, please cite the accompanying paper:
```
@misc{cryptoeprint:2024/714,
      author = {Shanxiang Lyu and Ling Liu and Cong Ling},
      title = {Learning With Quantization: A Ciphertext Efficient Lattice Problem with Tight Security Reduction from {LWE}},
      howpublished = {Cryptology {ePrint} Archive, Paper 2024/714},
      year = {2024},
      url = {[https://eprint.iacr.org/2024/714](https://eprint.iacr.org/2024/714)}
}
``` -->
## 7. License
This project is licensed under the MIT License.