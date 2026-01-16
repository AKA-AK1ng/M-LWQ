cmake_minimum_required(VERSION 3.10)

# [关键修改 1] 开启 ASM (汇编语言) 支持
# 必须在 C 后面加上 ASM，否则 CMake 无法识别 .S 文件
project(M_LWQ_C C ASM)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED True)

# 编译选项: 开启 AVX2, O3 优化, FMA
if(MSVC)
    add_compile_options(/arch:AVX2 /O2)
else()
    # 汇编文件也需要这些 flag (特别是 -mavx2)
    add_compile_options(-mavx2 -mfma -O3 -march=native -Wall -Wextra)
endif()

# 包含头文件路径
include_directories(src)
# [建议] 显式包含 avx2 目录，防止汇编找不到 consts.h
include_directories(src/avx2)
include_directories(src/avx512)

# 1. Common
file(GLOB COMMON_SRC "src/common/*.c")
add_library(mlwq_common STATIC ${COMMON_SRC})

# 2. Reference (Scalar)
file(GLOB REF_SRC "src/ref/*.c")
add_library(mlwq_ref STATIC ${REF_SRC})
target_link_libraries(mlwq_ref mlwq_common)

# 3. AVX2
# [关键修改 2] 修改 GLOB 模式，增加 .S 文件
# 这样 CMake 才会把 ntt.S, basemul.S 等汇编文件加入编译列表
file(GLOB AVX_SRC CONFIGURE_DEPENDS "src/avx2/*.c" "src/avx2/*.S")

add_library(mlwq_avx2 STATIC ${AVX_SRC})
target_link_libraries(mlwq_avx2 mlwq_common)

# 3. AVX512 (8-way SHAKE batching)
file(GLOB AVX512_SRC CONFIGURE_DEPENDS "src/avx512/*.c")
add_library(mlwq_avx512 STATIC ${AVX512_SRC})
target_link_libraries(mlwq_avx512 mlwq_common mlwq_avx2)
if(NOT MSVC)
    target_compile_options(mlwq_avx512 PRIVATE -mavx512f -mavx512dq -mavx512vl -mavx512bw)
endif()

# 4. Main Benchmark
add_executable(mlwq_bench src/main.c)
target_link_libraries(mlwq_bench mlwq_ref mlwq_avx2 mlwq_avx512 mlwq_common)

if(NOT WIN32)
    target_link_libraries(mlwq_bench m)
endif()

if(NOT MSVC)
    # -flto: 链接时优化，允许跨文件内联 (非常重要！)
    # -march=native: 允许 Ref 代码使用宿主机的指令集 (虽然不写汇编，但编译器会自动向量化)
    add_compile_options(-O3 -march=native -flto -fomit-frame-pointer)
    
    # LTO 需要在链接器也开启
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -flto")
endif()
