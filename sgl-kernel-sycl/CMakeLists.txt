cmake_minimum_required(VERSION 3.26 FATAL_ERROR)
project(sgl-kernel LANGUAGES CXX )
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl")
include(./dpctinstall/cmake/dpct.cmake)

# CMake
cmake_policy(SET CMP0169 OLD)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/utils.cmake)
set(CMAKE_COLOR_DIAGNOSTICS ON)
set(CMAKE_VERBOSE_MAKEFILE ON CACHE BOOL "ON")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_SHARED_LIBRARY_PREFIX "")

# Python
find_package(Python COMPONENTS Interpreter Development.Module ${SKBUILD_SABI_COMPONENT} REQUIRED)

# CXX
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# Torch
find_package(Torch REQUIRED)

# SYCL
enable_language(CXX)
find_package(IntelSYCL REQUIRED)
find_package(MKL REQUIRED)

message(STATUS "Detected COMPATIBILITY_VALUE=${COMPATIBILITY_VALUE}")
if ("${COMPATIBILITY_VALUE}" VERSION_GREATER_EQUAL "13.0")
    message("COMPATIBILITY_VALUE ${COMPATIBILITY_VALUE} >= 13.0")
elseif ("${COMPATIBILITY_VALUE}" VERSION_GREATER_EQUAL "12.8")
    message("COMPATIBILITY_VALUE ${COMPATIBILITY_VALUE} >= 12.8")
elseif ("${COMPATIBILITY_VALUE}" VERSION_GREATER_EQUAL "12.4")
    message("COMPATIBILITY_VALUE ${COMPATIBILITY_VALUE} >= 12.4")
elseif ("${COMPATIBILITY_VALUE}" VERSION_GREATER_EQUAL "12.1")
    message("COMPATIBILITY_VALUE ${COMPATIBILITY_VALUE} >= 12.1")
elseif ("${COMPATIBILITY_VALUE}" VERSION_GREATER_EQUAL "11.8")
    message("COMPATIBILITY_VALUE ${COMPATIBILITY_VALUE} >= 11.8")
endif()

message(STATUS "IntelSYCL include dirs: ${SYCL_INCLUDE_DIR}")

# clean Torch Flag
#clear_cuda_arches(CMAKE_FLAG)

include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/csrc
    /user/wanghao2/dles252install/compiler/2025.2/include
    /user/wanghao2/dles252install/compiler/2025.2/include/sycl
    ${PROJECT_SOURCE_DIR}/dpctinstall/dpct_header/include
    ${PROJECT_SOURCE_DIR}/build_sycl/_deps/repo-flashinfer-src/include
    /user/wanghao2/cutlass-syclinstall/cutlass-syclinstall/include
    /user/wanghao2/lzinstall/usr/include
    /user/wanghao2/lzdevinstall/usr/include
)

include_directories(${SYCL_INCLUDE_DIR})

link_directories(
    /user/wanghao2/lzinstall/usr/lib
    /user/wanghao2/lzdevinstall/usr/lib
)

set(SGL_KERNEL_CUDA_FLAGS
    "-DNDEBUG"
    "-DOPERATOR_NAMESPACE=sgl-kernel"
    "-O3"
    "-fPIC"
    "-std=c++17"
    "-DFLASHINFER_ENABLE_F16"
    "-DCUTE_USE_PACKED_TUPLE=1"
    "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1"
    "-DCUTLASS_VERSIONS_GENERATED"
    "-DCUTLASS_TEST_LEVEL=0"
    "-DCUTLASS_TEST_ENABLE_CACHED_RESULTS=1"
    "-DCUTLASS_DEBUG_TRACE_LEVEL=0"
    "-DCUTLASS_ENABLE_SYCL"
    # Suppress warnings
    "-Wconversion"
    "-fno-strict-aliasing"
    # uncomment to debug
    # "--ptxas-options=-v"
    # "--ptxas-options=--verbose,--register-usage-level=10,--warn-on-local-memory-usage"
)

option(SGL_KERNEL_ENABLE_SM100A           "Enable SM100A"           OFF)
option(SGL_KERNEL_ENABLE_SM90A            "Enable SM90A"            OFF)
option(SGL_KERNEL_ENABLE_BF16             "Enable BF16"             ON)
option(SGL_KERNEL_ENABLE_FP8              "Enable FP8"              ON)
option(SGL_KERNEL_ENABLE_FP4              "Enable FP4"              OFF)
option(SGL_KERNEL_ENABLE_FA3              "Enable FA3"              OFF)



list(APPEND SGL_KERNEL_CUDA_FLAGS
    "-ffast-math"
)


set(SOURCES
    "csrc/moe/moe_align_kernel.dp.cpp"
    "csrc/moe/moe_fused_gate.dp.cpp"
    "csrc/moe/moe_topk_softmax_kernels.dp.cpp"
    "csrc/moe/nvfp4_blockwise_moe.dp.cpp"
    "csrc/moe/fp8_blockwise_moe_kernel.dp.cpp"
    "csrc/moe/prepare_moe_input.dp.cpp"
    "csrc/moe/ep_moe_reorder_kernel.dp.cpp"
    "csrc/moe/ep_moe_silu_and_mul_kernel.dp.cpp"
    "csrc/moe/cutlass_moe/w4a8/scaled_mm_entry.dp.cpp"
    "csrc/moe/cutlass_moe/w4a8/w4a8_moe_data.dp.cpp"
    "csrc/moe/cutlass_moe/w4a8/w4a8_grouped_mm_c3x.dp.cpp"
    "csrc/common_extension.cc"
    "csrc/moe/marlin_moe_wna16/ops.dp.cpp"
    "csrc/moe/marlin_moe_wna16/gptq_marlin_repack.dp.cpp"
    "csrc/moe/marlin_moe_wna16/awq_marlin_repack.dp.cpp"
    "csrc/moe/marlin_moe_wna16/kernel_bf16_ku4.dp.cpp"
    "csrc/moe/marlin_moe_wna16/kernel_bf16_ku4b8.dp.cpp"
    "csrc/moe/marlin_moe_wna16/kernel_bf16_ku8b128.dp.cpp"
    "csrc/moe/marlin_moe_wna16/kernel_fp16_ku4.dp.cpp"
    "csrc/moe/marlin_moe_wna16/kernel_fp16_ku4b8.dp.cpp"
    "csrc/moe/marlin_moe_wna16/kernel_fp16_ku8b128.dp.cpp"
)

Python_add_library(common_ops MODULE USE_SABI ${SKBUILD_SABI_VERSION} WITH_SOABI ${SOURCES})

target_compile_options(common_ops PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${SGL_KERNEL_CUDA_FLAGS}>)
#target_include_directories(common_ops PRIVATE
#    ${repo-cutlass_SOURCE_DIR}/examples/77_blackwell_fmha
#    ${repo-cutlass_SOURCE_DIR}/examples/common
#    ${repo-flash-attention_SOURCE_DIR}/csrc/flash_attn/src
#)

find_package(Python3 COMPONENTS Interpreter REQUIRED)
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))"
    OUTPUT_VARIABLE TORCH_CXX11_ABI
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(TORCH_CXX11_ABI STREQUAL "0")
    message(STATUS "Using old C++ ABI (-D_GLIBCXX_USE_CXX11_ABI=0)")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
    set(CMAKE_SYCL_FLAGS "${CMAKE_SYCL_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
else()
    message(STATUS "Using new C++11 ABI (-D_GLIBCXX_USE_CXX11_ABI=1)")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=1")
    set(CMAKE_SYCL_FLAGS "${CMAKE_SYCL_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=1")
endif()

target_link_libraries(common_ops PRIVATE ${TORCH_LIBRARIES} c10  ${MKL_LIB})

install(TARGETS common_ops LIBRARY DESTINATION sgl_kernel_sycl)
