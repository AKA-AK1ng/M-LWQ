# MAMBA-Viper: High-Performance C Implementation with AVX2 Acceleration

This repository contains an optimized C implementation of the MAMBA-Viper public-key encryption and key-encapsulation scheme.
The design is built on the MLWQ-Z hardness foundation and includes both reference and AVX2-optimized implementations.
It features AVX2 SIMD acceleration and Number Theoretic Transform (NTT), targeting high-performance post-quantum cryptographic implementation and benchmarking.

**Contact:** make2024@stu2024.jnu.edu.cn

## 1. Core Concept

MAMBA-Viper is a lattice-based encryption and key-encapsulation design built on the MLWQ-Z framework.
At a high level, the scheme replaces conventional sampled error injection with a public quantization mechanism, aiming to combine compact communication, efficient implementation, and reduction-based security analysis.

Main design goals include:

- reduction-based security from the underlying MLWQ-Z problem
- compact public keys and ciphertexts through quantization-based compression
- high-performance implementation under NTT-friendly prime moduli and AVX2 acceleration

## 2. Implementation Features

### Optimization Highlights

- Vectorized arithmetic with explicit AVX2 intrinsics for core polynomial and module operations
- Fast NTT over \(R_q\) with \(n = 256\) and \(q = 3329\)
- Dual-mode benchmarking for reference C and AVX2 implementations
- Deterministic public-seed expansion for matrix generation and dither derivation

### Supported Algorithms

- `MAMBA-Viper.PKE` for IND-CPA public-key encryption
- `MAMBA-Viper.KEM` for IND-CCA2 key encapsulation via the Fujisaki-Okamoto transform

## 3. Build and Run

### Dependencies

- Compiler: GCC or Clang with C11 support
- Hardware: CPU with AVX2 support for the optimized implementation
- Build tool: CMake 3.10 or later

### Compilation

The `CMakeLists.txt` is configured to enable `-mavx2`, `-mfma`, and `-O3` for the optimized build.

```bash
git clone https://github.com/Make1205/MAMBA-Viper.git
cd M-LWQ

mkdir build
cd build

cmake ..
make
./mlwq_bench
```

# 4. Expected Output


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


>>> PART 2: Fair Pipeline Comparison (MLWQ Dither Path vs Kyber-like Error Path)
----------------------------------------------------------------------------------------------
Path       Mode       Sample/Gen         Add+Round          Total
----------------------------------------------------------------------------------------------
MLWQ       Scalar     11128              246                11374
MLWQ       AVX2       7090               134                7224
KyberEq    Scalar     3486               246                3732
KyberEq    AVX2       3370               134                3504

  MLWQ Total Speedup      : 1.57x
  KyberEq Total Speedup   : 1.06x
  Scalar Fairness Ratio   : 3.05x (MLWQ/KyberEq)
  AVX2 Fairness Ratio     : 2.06x (MLWQ/KyberEq)


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
## 6. Academic Citation and Provenance

This repository contains the implementation accompanying the following paper:

```bibtex
@misc{cryptoeprint:2024/714,
  author       = {Shanxiang Lyu and Ling Liu and Cong Ling},
  title        = {Learning With Quantization: A Ciphertext Efficient Lattice Problem with Tight Security Reduction from {LWE}},
  howpublished = {Cryptology {ePrint} Archive, Paper 2024/714},
  year         = {2024},
  url          = {https://eprint.iacr.org/2024/714}
}
```
<!-- ## 7. License
This project is licensed under the MIT License. -->
