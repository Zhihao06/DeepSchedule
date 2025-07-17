# NOTES: this CMake is only for debugging; for setup, please use Torch extension
cmake_minimum_required(VERSION 3.10)
project(deep_ep LANGUAGES CUDA CXX)
set(CMAKE_VERBOSE_MAKEFILE ON)

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

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -fPIC -fvisibility=default")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fPIC -fvisibility=default -DDISABLE_AGGRESSIVE_PTX_INSTRS")
set(CUDA_SEPARABLE_COMPILATION ON)
set(NVSHMEM_ROOT_DIR "/home/dpsk_a2a/deepep-nvshmem/install")
list(APPEND CUDA_NVCC_FLAGS "-O3")
list(APPEND CUDA_NVCC_FLAGS
    -Xcompiler
    -O3
    -rdc=true
    --ptxas-options=--register-usage-level=10
    -DDISABLE_AGGRESSIVE_PTX_INSTRS
)

set(TORCH_CUDA_ARCH_LIST "9.0")
find_package(CUDAToolkit REQUIRED)
find_package(pybind11 REQUIRED)
set(Torch_DIR "/mnt/data/nas/zhihao/zhl-sglang/lib/python3.10/site-packages/torch/share/cmake/Torch")
find_package(Torch REQUIRED)
find_package(NVSHMEM REQUIRED HINTS ${NVSHMEM_ROOT_DIR}/lib/cmake/nvshmem)

add_library(nvshmem ALIAS nvshmem::nvshmem)
add_library(nvshmem_host ALIAS nvshmem::nvshmem_host)
add_library(nvshmem_device ALIAS nvshmem::nvshmem_device)

# Seems bugs with CMake, NVCC 12 and C++ 17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)

include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include ${NVSHMEM_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS})
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib ${NVSHMEM_LIB_DIR} ${TORCH_INSTALL_PREFIX}/lib)

add_subdirectory(kernels)

set(SOURCES deep_ep.cpp)
add_library(deep_ep SHARED ${SOURCES})
set_property(TARGET deep_ep PROPERTY CUDA_ARCHITECTURES 90)
target_link_libraries(deep_ep PRIVATE CUDA::cudart ${EP_CUDA_LIBRARIES} ${TORCH_LIBRARIES} torch_python)

# Link CPP and CUDA together
# pybind11_add_module(deep_ep_cpp deep_ep.cpp)
# target_link_libraries(deep_ep_cpp PRIVATE ${EP_CUDA_LIBRARIES}) # ${TORCH_LIBRARIES} torch_python
