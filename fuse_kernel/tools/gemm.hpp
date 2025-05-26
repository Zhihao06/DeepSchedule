#pragma once

#include "cutlass/cutlass.h"
#include "deep_gemm/fp8_gemm.cuh"

using namespace deep_gemm;

void launch_gemm(void* __raw_lhs, void* __raw_lhs_scales, void* __raw_rhs, void* __raw_rhs_scales, void* __raw_out, void* __raw_grouped_layout, int m, void* __raw_stream) {
    int num_sms = 78;
    int smem_size = 206800;
    // Cast raw types (if needed)
    auto lhs = reinterpret_cast<__nv_fp8_e4m3*>(__raw_lhs);
    auto lhs_scales = reinterpret_cast<float*>(__raw_lhs_scales);
    auto rhs = reinterpret_cast<__nv_fp8_e4m3*>(__raw_rhs);
    auto rhs_scales = reinterpret_cast<float*>(__raw_rhs_scales);
    auto out = reinterpret_cast<__nv_bfloat16*>(__raw_out);
    auto grouped_layout = reinterpret_cast<int*>(__raw_grouped_layout);
    auto stream = reinterpret_cast<cudaStream_t>(__raw_stream);

    using namespace deep_gemm;

    // Templated args from Python JIT call
    constexpr auto N = 4096, K = 1536;
    constexpr auto BLOCK_M = 64;
    constexpr auto BLOCK_N = 144;
    constexpr auto BLOCK_K = 128;
    constexpr auto BLOCK_N_PADDING = 0;
    constexpr auto kSwizzleDMode = 32;
    constexpr auto kNumGroups = 16;
    constexpr auto kNumStages = 7;
    constexpr auto kNumTMAMulticast = 1;
    constexpr auto kIsTMAMulticastOnA = true;

    // Make a templated grouped GEMM
    using gemm_t = Gemm<N, K, BLOCK_M, BLOCK_N, BLOCK_K, BLOCK_N_PADDING, kSwizzleDMode, kNumGroups, kNumStages, kNumTMAMulticast, kIsTMAMulticastOnA, GemmType::GroupedMasked>;

    // Launch kernel
    auto tma_a_desc = gemm_t::make_2d_tma_a_desc(lhs, m);
    auto tma_b_desc = gemm_t::make_2d_tma_b_desc(rhs);
    auto tma_scales_a_desc = gemm_t::make_2d_tma_scales_a_desc(lhs_scales, m);
    auto tma_d_desc = gemm_t::make_2d_tma_d_desc(out, m);
    gemm_t::run(out, rhs_scales, grouped_layout,
                m,
                tma_a_desc, tma_b_desc, tma_scales_a_desc, tma_d_desc,
                stream, num_sms, smem_size);
}
