# M-LWQ-C: High-Performance C Implementation with AVX2 Acceleration
This repository contains an optimized C implementation of the M-LWQ (Module-Learning With Quantization) Public Key Encryption (PKE) scheme and Key Encapsulation Mechanism (KEM).
It features AVX2 SIMD acceleration and Number Theoretic Transform (NTT), providing high-performance post-quantum cryptographic primitives with tight security reductions.
## 1. Core Concept (M-LWQ)
 M-LWQ is a novel post-quantum cryptosystem built on the Learning With Quantization (LWQ) problem. It replaces the randomized error sampling of LWE with a deterministic dithered quantization process achieving:
 - Tight Security: A tight security reduction from the standard Module-LWE problem.
 - Bandwidth Efficiency: State-of-the-art compactness, achieving approximately 20% smaller public keys compared to Kyber-512.
 Algorithmic Simplicity: Eliminates the need for high-precision Gaussian sampling by using compensated dithering.
 
## 2. Implementation Features

 🚀 Optimization Highlights
 - Vectorized Arithmetic: Explicit AVX2 Intrinsics for polynomial operations, achieving over 2.5x speedup in core matrix-vector multiplications.
 - Fast NTT: Optimized Number Theoretic Transform for $R_q$ with $n=256, q=3329$, utilizing Barrett Reduction for rapid modular arithmetic.
 - Dual-Mode Benchmark: Built-in suite to compare Scalar C vs. AVX2 performance in real-time.
 - Zero-Bandwidth Dithering: Dither vectors are derived deterministically from public seeds to maintain compactness.

 🛠 Supported Algorithms
 M-LWQ.PKE: IND-CPA secure Public Key Encryption.
 M-LWQ.KEM: IND-CCA2 secure Key Encapsulation Mechanism via the Fujisaki-Okamoto transform.

## 3. Build and Run
Dependencies
 - Compiler: GCC or Clang (supporting C11).
 - Hardware: CPU with AVX2 support.
 - Build Tool: CMake 3.10+.
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
./mlwq_bench

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

----------------------------------------------------------------------------------------------
 PKE KeyGen Breakdown (Detailed)
----------------------------------------------------------------------------------------------
Sub-Component        Scalar (cyc)    AVX2 (cyc)      Speedup         Scalar %  
GenMatrix (A)        21436           8028            2.67x            31.1%
Sample (s)           3486            3370            1.03x            5.1%
GenDither            11128           7090            1.57x            16.1%
Arith (A*s)          27668           9370            2.95x            40.1%
Quantize             246             134             1.83x            0.4%

----------------------------------------------------------------------------------------------
 PKE Encrypt Breakdown (Detailed)
----------------------------------------------------------------------------------------------
Sub-Component        Scalar (cyc)    AVX2 (cyc)      Speedup        
Arith (u)            32612           12057           2.70x
Arith (v)            21172           8372            2.53x

----------------------------------------------------------------------------------------------
 PKE Decrypt Breakdown (Detailed)
----------------------------------------------------------------------------------------------
Sub-Component        Scalar (cyc)    AVX2 (cyc)      Speedup        
DeQuantize           3972            4121            0.96x
Arith (v-su)         27668           9370            2.95x
Decode               3662            3529            1.04x


>>> PART 2: Core Component Comparison (Quantize vs Sample)
----------------------------------------------------------------------------------------------
Component  Mode       Quantize        Sample          Alg. Efficiency      AVX Improvement     
----------------------------------------------------------------------------------------------
PK / u     Scalar     246             3486            14.17x                 1.00x (Ref)         
PK / u     AVX2       134             3370            25.15x                 1.84x


>>> PART 3: PKE Full Flow Summary (Total Time)
----------------------------------------------------------------------------------------------
Operation            Scalar Cycles   AVX2 Cycles     Speedup   
----------------------------------------------------------------------------------------------
PKE KeyGen           68951           32902           2.10x
PKE Encrypt          90575           38061           2.38x
PKE Decrypt          29822           13105           2.28x


>>> PART 4: KEM Full Flow Summary (IND-CCA2)
----------------------------------------------------------------------------------------------
Operation            Scalar Cycles   AVX2 Cycles     Speedup   
----------------------------------------------------------------------------------------------
KEM KeyGen           79774           40506           1.97x
KEM Encaps           106101          51202           2.07x
KEM Decaps           119020          48285           2.46x
----------------------------------------------------------------------------------------------

[FINAL] All checks passed! Implementation is correct.
```
Note: Speedup factors depend on your specific CPU architecture. The NTT implementation reduces complexity from quadratic to log-linear, providing significant gains even without AVX, while AVX2 further accelerates the vectorized operations.



## 5. Project Structure
```
M_LWQ_v1_c_imple/
├── CMakeLists.txt          # CMake build configuration file
├── Readme.md               # Project documentation
└── src/                    # Main source code directory
    ├── main.c              # Entry point / Benchmark code
    ├── common/             # Common utilities (Hash, Random, Params)
    │   ├── cycles.h        # CPU cycle counter (for benchmarking)
    │   ├── fips202.c/h     # Scalar SHAKE/Keccak implementation
    │   ├── params.h        # M-LWQ parameter definitions (N, Q, K, etc.)
    │   ├── random.c/h      # Random number generator
    │   └── structs.h       # Data structure definitions
    │
    ├── ref/                # Pure C Reference Implementation
    │   ├── mlwq.c/h        # Core logic: KeyGen, Encrypt, Decrypt
    │   ├── ntt.c/h         # Number Theoretic Transform (NTT) reference implementation
    │   ├── poly.c/h        # Polynomial operations (arithmetic, sampling, packing)
    │   ├── reduce.h        # Modular reduction functions
    │   └── xof.c/h         # Extensible Output Functions (Matrix/Vector expansion)
    │
    └── avx2/               # AVX2 Instruction Set Optimization
        ├── consts.c/h      # AVX2 pre-computed constants (qdata, shuffle masks, etc.)
        ├── fips202x4.c/h   # Interface for 4-way parallel SHAKE/Keccak
        ├── mlwq.c/h        # AVX2 optimized core logic
        ├── ntt.c/h         # AVX2 optimized NTT/InvNTT and pointwise multiplication
        ├── poly.c/h        # AVX2 optimized polynomial operations and packing
        ├── xof.c/h         # AVX2 optimized parallel Matrix/Vector sampling
        │
        └── keccak4x/       # 4-way parallel Keccak core (Assembly/Intrinsics)
            ├── KeccakP-1600-times4-SIMD256.c
            ├── KeccakP-1600-times4-SnP.h
            ├── KeccakP-1600-unrolling.macros
            ├── KeccakP-align.h
            ├── KeccakP-brg_endian.h
            └── KeccakP-SIMD256-config.h
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
<!-- ## 7. License
This project is licensed under the MIT License. -->