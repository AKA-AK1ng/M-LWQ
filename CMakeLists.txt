cmake_minimum_required(VERSION 3.10)
project(M_LWQ_PKE CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 开启 AVX2 和 FMA 优化
if(MSVC)
    add_compile_options(/arch:AVX2 /O2)
else()
    add_compile_options(-mavx2 -mfma -O3 -march=native -Wall -Wextra)
endif()

include_directories(src)

# 1. 编译 Common 库 (公共组件)
file(GLOB COMMON_SRC "src/common/*.cpp")
add_library(mlwq_common STATIC ${COMMON_SRC})

# 2. 编译 Reference 库 (纯 C++)
file(GLOB REF_SRC "src/ref/*.cpp")
add_library(mlwq_ref STATIC ${REF_SRC})
target_link_libraries(mlwq_ref mlwq_common)

# 3. 编译 AVX2 库 (Intrinsics)
# 注意：包含 keccak4x
file(GLOB AVX_SRC "src/avx2/*.cpp")
add_library(mlwq_avx2 STATIC ${AVX_SRC})
target_link_libraries(mlwq_avx2 mlwq_common)

# 4. 编译主程序
add_executable(mlwq_bench src/main.cpp)
target_link_libraries(mlwq_bench mlwq_ref mlwq_avx2 mlwq_common)

if(NOT WIN32)
    target_link_libraries(mlwq_bench m)
endif()