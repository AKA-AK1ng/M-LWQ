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

# STM32 HAL 宏定义 (使 stm32f4xx_hal.h 和 random.c 正确展开)
add_compile_definitions(
    STM32F407xx
    USE_HAL_DRIVER
)

# STM32 HAL 头文件路径 (根据 CubeMX 生成的 Drivers 目录结构)
# 若路径不同，可在调用 cmake 时覆盖: -DHAL_DRIVER_DIR=<path>
if(NOT DEFINED HAL_DRIVER_DIR)
    set(HAL_DRIVER_DIR "${CMAKE_SOURCE_DIR}/Drivers")
endif()

include_directories(
    src
    ${HAL_DRIVER_DIR}/STM32F4xx_HAL_Driver/Inc
    ${HAL_DRIVER_DIR}/CMSIS/Device/ST/STM32F4xx/Include
    ${HAL_DRIVER_DIR}/CMSIS/Include
)

# 启动文件 (CubeMX 生成, 负责向量表/堆栈初始化)
set(STARTUP_FILE "${CMAKE_SOURCE_DIR}/startup_stm32f407xx.s")

# 链接脚本 (CubeMX 生成, 定义 FLASH/RAM 内存布局)
set(LINKER_SCRIPT "${CMAKE_SOURCE_DIR}/STM32F407ZGTx_FLASH.ld")

# 1. Common
file(GLOB COMMON_SRC "src/common/*.c")
add_library(mlwq_common STATIC ${COMMON_SRC})

# 2. Reference (Scalar) — 仅使用可移植的标量实现，不使用 AVX2
file(GLOB REF_SRC "src/ref/*.c")
add_library(mlwq_ref STATIC ${REF_SRC})
target_link_libraries(mlwq_ref mlwq_common)

# 3. AVX2 已禁用 — AVX2 为 x86 专属指令集，不支持 ARM Cortex-M4

# 4. Main Benchmark (包含启动文件以生成可烧录的 ELF 固件)
add_executable(mlwq_bench src/main.c ${STARTUP_FILE})
target_link_libraries(mlwq_bench mlwq_ref mlwq_common)

# 链接器: 指定 STM32 内存布局脚本，去除未使用的段以减小固件体积
set(CMAKE_EXE_LINKER_FLAGS
    "${CMAKE_EXE_LINKER_FLAGS} -T${LINKER_SCRIPT} -Wl,--gc-sections"
)