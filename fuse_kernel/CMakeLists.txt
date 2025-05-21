# NOTES: current just for CMake-based IDE (e.g. CLion) indexing, the real compilation is done via JIT
# TODO: add CUDA utils' library via CMake
cmake_minimum_required(VERSION 3.10)
project(deep_gemm LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(NVSHMEM_ROOT_DIR "/home/dpsk_a2a/deepep-nvshmem/install")
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")

find_package(CUDAToolkit REQUIRED)
find_package(Python REQUIRED)
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 REQUIRED)
find_package(NVSHMEM REQUIRED HINTS ${NVSHMEM_ROOT_DIR}/lib/cmake/nvshmem)

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

add_library(nvshmem ALIAS nvshmem::nvshmem)
add_library(nvshmem_host ALIAS nvshmem::nvshmem_host)
add_library(nvshmem_device ALIAS nvshmem::nvshmem_device)

include_directories(../third-party/cutlass/include ../third-party/cutlass/tools/util/include)
include_directories(../deep_gemm/include)
include_directories(../deep_ep/csrc)
include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include ${NVSHMEM_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS})
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib ${NVSHMEM_LIB_DIR} ${TORCH_INSTALL_PREFIX}/lib)
link_directories(../deep_ep/csrc/build)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fPIC")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -fPIC -DNDEBUG")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 -std=c++17 -DNDEBUG --ptxas-options=--register-usage-level=10")

# add_executable(fuse_app fuse.cu)
add_executable(ep ep.cpp)
target_link_libraries(ep PUBLIC deep_ep nvshmem cudart cudadevrt mlx5 CUDA::cudart ${PYTHON_LIBRARIES} ${TORCH_LIBRARIES} torch_python)
