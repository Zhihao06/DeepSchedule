cmake_minimum_required(VERSION 3.10)
project(deep_fuse LANGUAGES CXX CUDA)

add_subdirectory(deep_gemm/include)
add_subdirectory(deep_ep/csrc)
