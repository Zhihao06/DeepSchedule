function(add_op_library target_name source_file)
    add_library(${target_name} STATIC ${source_file})
    set_target_properties(${target_name} PROPERTIES
            POSITION_INDEPENDENT_CODE ON
            CXX_STANDARD_REQUIRED ON
            CUDA_STANDARD_REQUIRED ON
            CXX_STANDARD 20
            CUDA_STANDARD 20
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_ARCHITECTURES 90
    )
    target_include_directories(${target_name} PUBLIC ${CUDA_TOOLKIT_ROOT_DIR}/include ${TORCH_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS})
    target_link_libraries(${target_name} PUBLIC cudart cudadevrt ${PYTHON_LIBRARIES} ${TORCH_LIBRARIES})
endfunction()

add_op_library(fuse_silu_and_mul fuse_silu_and_mul.cu)
