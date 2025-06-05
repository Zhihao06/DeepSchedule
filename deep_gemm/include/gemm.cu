#include "cutlass/cutlass.h"
#include "deep_gemm/fp8_gemm.cuh"
#include "gemm.cuh"

template <uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t BLOCK_N_PADDING,
          uint32_t kSwizzleDMode,
          uint32_t kNumGroups, uint32_t kNumStages,
          uint32_t kNumTMAMulticast, bool kIsTMAMulticastOnA,
          uint32_t SMEM_SIZE>
void launch_gemm(void* __raw_lhs, void* __raw_lhs_scales, void* __raw_rhs, void* __raw_rhs_scales, void* __raw_out, void* __raw_grouped_layout, int m, void* __raw_stream, int num_sms) {
    // int num_sms = 78;
    int smem_size = SMEM_SIZE;
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
    constexpr auto N = SHAPE_N, K = SHAPE_K;
    // constexpr auto BLOCK_M = BLOCK_M;
    // constexpr auto BLOCK_N = BLOCK_N;
    // constexpr auto BLOCK_K = BLOCK_K;
    // constexpr auto BLOCK_N_PADDING = BLOCK_N_PADDING;
    // constexpr auto kSwizzleDMode = kSwizzleDMode;
    // constexpr auto kNumGroups = kNumGroups;
    // constexpr auto kNumStages = kNumStages;
    // constexpr auto kNumTMAMulticast = kNumTMAMulticast;
    // constexpr auto kIsTMAMulticastOnA = kIsTMAMulticastOnA;

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
