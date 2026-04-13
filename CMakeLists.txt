cmake_minimum_required(VERSION 3.10)

project(M_LWQ_C C ASM)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED True)

# 目标平台: STM32F407ZGT6 (ARM Cortex-M4F)
# 使用 arm-none-eabi-gcc 交叉编译工具链
# 示例: cmake -DCMAKE_TOOLCHAIN_FILE=<toolchain.cmake> ..
# 或者直接设置: cmake -DCMAKE_C_COMPILER=arm-none-eabi-gcc ..

# STM32F407ZGT6 编译选项 (Cortex-M4, 硬件浮点, Thumb-2)
# AVX2/FMA/march=native 均不适用于 ARM 平台，已全部禁用
add_compile_options(
    -mcpu=cortex-m4
    -mthumb
    -mfpu=fpv4-sp-d16
    -mfloat-abi=hard
    -O2
    -Wall
    -Wextra
    -ffunction-sections
    -fdata-sections
)

# 包含头文件路径
include_directories(src)

# 1. Common
file(GLOB COMMON_SRC "src/common/*.c")
add_library(mlwq_common STATIC ${COMMON_SRC})

# 2. Reference (Scalar) — 仅使用可移植的标量实现，不使用 AVX2
file(GLOB REF_SRC "src/ref/*.c")
add_library(mlwq_ref STATIC ${REF_SRC})
target_link_libraries(mlwq_ref mlwq_common)

# 3. AVX2 已禁用 — AVX2 为 x86 专属指令集，不支持 ARM Cortex-M4

# 4. Main Benchmark
add_executable(mlwq_bench src/main.c)
target_link_libraries(mlwq_bench mlwq_ref mlwq_common)

# 链接器: 去除未使用的段以减小固件体积
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--gc-sections")