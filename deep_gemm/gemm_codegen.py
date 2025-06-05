from tools.gemm_config_tuner import get_best_configs
from deep_gemm import get_num_sms
from typing import List

code_gemm_cu = """#include "cutlass/cutlass.h"
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
"""

code_gemm_kernel = """#include "cutlass/cutlass.h"
#include "deep_gemm/fp8_gemm.cuh"
#include "gemm.cu"
"""

code_gemm_gen_hpp = """
#pragma once

#include <stdexcept>
#include "gemm.cuh"

using GemmLauncherFunc = void (*)(void*, void*, void*, void*, void*, void*, int, void*, int);

GemmLauncherFunc get_function_for_gemm(int num_tokens, int hidden, int intermediate, int num_groups, int sms) {"""

code_gemm_end = """
    else {
        throw std::runtime_error("Unsupported parameters for gemm: num_tokens=" + 
                                std::to_string(num_tokens) + ", hidden=" + 
                                std::to_string(hidden) + ", intermediate=" + 
                                std::to_string(intermediate) + ", num_groups=" +
                                std::to_string(num_groups) + ", sms=" +
                                std::to_string(sms));
    }
}
"""

gemm_gen_branch = lambda num_tokens, hidden, intermediate, num_groups, sms, is_first, SHAPE_N, SHAPE_K, BLOCK_M, BLOCK_N, BLOCK_K, BLOCK_N_PADDING, kSwizzleDMode, kNumGroups, kNumStages, kNumTMAMulticast, kIsTMAMulticastOnA, SMEM_SIZE: f"""
    {"if" if is_first else "else if"} (num_tokens == {num_tokens} and hidden == {hidden} and intermediate == {intermediate} and num_groups == {num_groups} and sms == {sms}) return launch_gemm<{SHAPE_N}, {SHAPE_K}, {BLOCK_M}, {BLOCK_N}, {BLOCK_K}, {BLOCK_N_PADDING}, {kSwizzleDMode}, {kNumGroups}, {kNumStages}, {kNumTMAMulticast}, {"true" if kIsTMAMulticastOnA else "false"}, {SMEM_SIZE}>;"""

gemm_cu_branch = lambda SHAPE_N, SHAPE_K, BLOCK_M, BLOCK_N, BLOCK_K, BLOCK_N_PADDING, kSwizzleDMode, kNumGroups, kNumStages, kNumTMAMulticast, kIsTMAMulticastOnA, SMEM_SIZE: f"""
template void launch_gemm<{SHAPE_N}, {SHAPE_K}, {BLOCK_M}, {BLOCK_N}, {BLOCK_K}, {BLOCK_N_PADDING}, {kSwizzleDMode}, {kNumGroups}, {kNumStages}, {kNumTMAMulticast}, {"true" if kIsTMAMulticastOnA else "false"}, {SMEM_SIZE}>(void* __raw_lhs, void* __raw_lhs_scales, void* __raw_rhs, void* __raw_rhs_scales, void* __raw_out, void* __raw_grouped_layout, int m, void* __raw_stream, int num_sms);"""

def generate_kernel_template(num_tokens_l: List, hidden_l: List, intermediate_l: List, num_groups_l: List, sms_l: List):
    global code_gemm_cu, code_gemm_gen_hpp, code_gemm_end
    is_first = True
    gemm_cu_set = set()
    for num_tokens in num_tokens_l:
        expected_m = max(1, (num_tokens * 8 + 128 - 1) / 128 * 2)
        for hidden in hidden_l:
            for intermediate in intermediate_l:
                for num_groups in num_groups_l:
                    for sms in sms_l:
                        n = intermediate
                        k = hidden
                        num_sms, block_m, block_n, num_stages, tma_multicast_config, smem_config, num_waves = get_best_configs(expected_m, n, k, num_groups, sms, is_grouped_masked=True)
                        code_gemm_gen_hpp += gemm_gen_branch(num_tokens, n, k, num_groups, sms, is_first, n, k, block_m, block_n, 128, smem_config[2], smem_config[1], num_groups, num_stages, tma_multicast_config[0], tma_multicast_config[1], smem_config[0])
                        gemm_cu_set.add((n, k, block_m, block_n, 128, smem_config[2], smem_config[1], num_groups, num_stages, tma_multicast_config[0], tma_multicast_config[1], smem_config[0]))
                        print(
                            f"SHAPE_N, SHAPE_K, BLOCK_M, BLOCK_N, BLOCK_K, BLOCK_N_PADDING, kSwizzleDMode, kNumGroups, kNumStages, kNumTMAMulticast, kIsTMAMulticastOnA, SMEM_SIZE>\n"
                            f"<{n}, {k}, {block_m}, {block_n}, {128}, {smem_config[2]}, {smem_config[1]}, {num_groups}, {num_stages}, {tma_multicast_config[0]}, {tma_multicast_config[1]}, {smem_config[0]}>\n"
                            f"num_sms={num_sms}, block_m={block_m}, block_n={block_n}, num_stages={num_stages}, tma_multicast_config={tma_multicast_config}, smem_config={smem_config}"
                        )
                        is_first = False

                        n = hidden
                        k = int(intermediate/2)
                        num_sms, block_m, block_n, num_stages, tma_multicast_config, smem_config, num_waves = get_best_configs(expected_m, n, k, num_groups, sms, is_grouped_masked=True)
                        code_gemm_gen_hpp += gemm_gen_branch(num_tokens, n, k, num_groups, sms, is_first, n, k, block_m, block_n, 128, smem_config[2], smem_config[1], num_groups, num_stages, tma_multicast_config[0], tma_multicast_config[1], smem_config[0])
                        gemm_cu_set.add((n, k, block_m, block_n, 128, smem_config[2], smem_config[1], num_groups, num_stages, tma_multicast_config[0], tma_multicast_config[1], smem_config[0]))

    for gemm_cu_config in gemm_cu_set:
        text = ""
        text += code_gemm_kernel
        text += gemm_cu_branch(*gemm_cu_config)
        file_name = "kernels/kernel.m_grouped_gemm_fp8_fp8_bf16_nt." + "_".join(map(str, gemm_cu_config)) + ".cu"
        with open(file_name, "w") as f:
            f.write(text)

    code_gemm_gen_hpp += code_gemm_end
    with open("../fuse_kernel/tools/gemm_gen.hpp", "w") as f:
        f.write(code_gemm_gen_hpp)

def test_best_configs():
    expected_m, n, k, num_groups, sms = 16, 4096, 1536, 16, 50
    num_sms, block_m, block_n, num_stages, tma_multicast_config, smem_config, num_waves = get_best_configs(expected_m, n, k, num_groups, sms, is_grouped_masked=True)
    print(
        f"SHAPE_N, SHAPE_K, BLOCK_M, BLOCK_N, BLOCK_K, BLOCK_N_PADDING, kSwizzleDMode, kNumGroups, kNumStages, kNumTMAMulticast, kIsTMAMulticastOnA, SMEM_SIZE>\n"
        f"<{n}, {k}, {block_m}, {block_n}, {128}, {smem_config[2]}, {smem_config[1]}, {num_groups}, {num_stages}, {tma_multicast_config[0]}, {tma_multicast_config[1]}, {smem_config[0]}>\n"
        f"num_sms={num_sms}, block_m={block_m}, block_n={block_n}, num_stages={num_stages}, tma_multicast_config={tma_multicast_config}, smem_config={smem_config}, num_waves={num_waves}"
    )

if __name__ == "__main__":
    generate_kernel_template(
        [32], # num_tokens_l , 64, 128, 256
        [4096], # hidden_l
        [3072], # intermediate_l
        [16], # num_groups_l
        [54] # sms_l
    )

    # generate_kernel_template(
    #     [32, 64, 128, 256, 512], # num_tokens_l
    #     [2048, 4096, 8192, 16384], # hidden_l
    #     [2048, 3072, 4096, 8192, 16384], # intermediate_l
    #     [4, 8, 16, 32], # num_groups_l
    #     range(6, 66, 4) # sms_l
    # )
    # test_best_configs()