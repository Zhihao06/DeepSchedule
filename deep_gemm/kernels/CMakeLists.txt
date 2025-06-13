# NOTES: current just for CMake-based IDE (e.g. CLion) indexing, the real compilation is done via JIT
# TODO: add CUDA utils' library via CMake
cmake_minimum_required(VERSION 3.10)
project(deep_gemm_kernels LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")

# 查找 ccache
find_program(CCACHE_PROGRAM NAMES ccache)

if(CCACHE_PROGRAM)
    message(STATUS "Found ccache: ${CCACHE_PROGRAM}")
    
    # 设置 C/CXX 编译器为 ccache 包裹的版本
    set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")

    # 如果使用 nvcc 编译 CUDA 代码，也用 ccache 包装
    if(CMAKE_CUDA_COMPILER)
        set(CMAKE_CUDA_HOST_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
        set(CMAKE_CUDA_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    endif()
else()
    message(STATUS "ccache not found, compiling without ccache")
endif()

find_package(CUDAToolkit REQUIRED)
# find_package(Python REQUIRED)
# set(PYTHON_INCLUDE_DIRS "/usr/include/python3.10")
# set(PYTHON_LIBRARIES "/usr/lib/x86_64-linux-gnu/libpython3.10.so")
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 REQUIRED)

file(WRITE ${CMAKE_BINARY_DIR}/test_cuda.cu "extern \"C\" __global__ void testKernel() { }")
execute_process(
        COMMAND ${CUDA_NVCC_EXECUTABLE} ${CMAKE_CUDA_FLAGS} -gencode arch=compute_90a,code=sm_90a -o ${CMAKE_BINARY_DIR}/test_cuda.o -c ${CMAKE_BINARY_DIR}/test_cuda.cu
        RESULT_VARIABLE NVCC_RESULT
        OUTPUT_VARIABLE NVCC_OUTPUT
        ERROR_VARIABLE NVCC_ERROR_OUTPUT
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

if (NVCC_RESULT EQUAL "0")
    set(NVCC_SUPPORTS_SM90 TRUE)
    message(STATUS "NVCC supports SM90")
else()
    message(STATUS "NVCC does not support SM90")
endif()

if (NVCC_SUPPORTS_SM90)
    set(TORCH_CUDA_ARCH_LIST "8.6" CACHE STRING "Add arch tag 90a to NVCC" FORCE)
    list(APPEND CUDA_NVCC_FLAGS "-gencode;arch=compute_90a,code=sm_90a")
endif()
set(Torch_DIR "/usr/local/lib/python3.10/dist-packages/torch/share/cmake/Torch")
find_package(Torch REQUIRED)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fPIC")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -fPIC -DNDEBUG")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 -std=c++17 -DNDEBUG --ptxas-options=--register-usage-level=10")

include_directories("../include" ${CUDA_TOOLKIT_ROOT_DIR}/include ${TORCH_INCLUDE_DIRS} ../../third-party/cutlass/include ../../third-party/cutlass/tools/util/include)

file(GLOB CU_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cu")

# debug时可以根据gemm_code_gen.py生成内容可以只编译一些cu文件来加快编译速度
# set(CU_FILES
#     kernel.m_grouped_gemm_fp8_fp8_bf16_nt.3072_4096_64_160_128_0_64_32_4_1_True_136512.cu
#     kernel.m_grouped_gemm_fp8_fp8_bf16_nt.4096_1536_64_160_128_0_64_16_4_1_True_136352.cu
#     kernel.m_grouped_gemm_fp8_fp8_bf16_nt.3072_4096_64_160_128_0_64_16_4_1_True_136512.cu
#     kernel.m_grouped_gemm_fp8_fp8_bf16_nt.4096_1536_64_160_128_0_64_32_4_1_True_136352.cu
# )

# 静态链接
add_library(all_kernels_obj OBJECT ${CU_FILES})
target_link_libraries(all_kernels_obj PRIVATE
    CUDA::cudart
    cudadevrt
    ${TORCH_LIBRARIES}
)
add_library(deep_gemm_kernels STATIC $<TARGET_OBJECTS:all_kernels_obj>)

# 动态链接
# foreach(CU_FILE ${CU_FILES})
#     string(REGEX REPLACE "\\.cu$" "" BASE_NAME "${CU_FILE}")
#     message(STATUS "${BASE_NAME}")
#     set(SRC_FILE "${CMAKE_CURRENT_SOURCE_DIR}/${CU_FILE}")
#     add_library(${BASE_NAME} SHARED ${SRC_FILE})
#     set_target_properties(${BASE_NAME} PROPERTIES
#         PREFIX ""
#         OUTPUT_NAME ${BASE_NAME}
#         LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib
#     )
#     target_link_libraries(${BASE_NAME} PRIVATE
#         CUDA::cudart
#         cudadevrt
#         ${TORCH_LIBRARIES}
#     )
# endforeach()

# file(GLOB SO_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "${CMAKE_CURRENT_SOURCE_DIR}/lib/*.so")
# message(STATUS "${SO_FILES}")
