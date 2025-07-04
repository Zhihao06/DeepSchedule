#pragma once

#include "base_moe.hpp"

using namespace deep_ep;
using namespace c10d;

class SequenceMoE : public BaseMoE {
protected:
    void _moe_core(std::shared_ptr<FUSEConfig>& fuse_config, bool enable_profile) {
        cudaStream_t current_stream = at::cuda::getCurrentCUDAStream();
        if (enable_profile) cudaProfilerStart();
        global_pg->barrier()->wait();
        auto [
            packed_recv_x, 
            packed_recv_x_scales, 
            packed_recv_count, 
            packed_recv_src_info, 
            packed_recv_layout_range, 
            event, 
            hook
        ] = buffer->low_latency_dispatch(hidden_states, topk_ids, num_max_dispatch_tokens_per_rank, num_experts, fuse_config->ep_sms, true/*use_fp8*/, false/*async_finish*/, false/*return_recv_hook*/);
        get_function_for_gemm(num_tokens, khidden, hidden_size, num_groups, fuse_config->gemm_sms)(
            std::get<0>(x_fp8).data_ptr(), std::get<1>(x_fp8).data_ptr(),
            std::get<0>(y_fp8).data_ptr(), std::get<1>(y_fp8).data_ptr(),
            out.data_ptr(),
            packed_recv_count.data_ptr(),
            expected_m,
            current_stream,
            fuse_config->gemm_sms
        );
        auto out_size = static_cast<int64_t>(num_groups * m_max);
        auto out_view = out.view({out_size, -1});
        fuse_silu_and_mul_masked(silu_out, o_vec, o_scales, out_view, packed_recv_count, m_max);
        get_function_for_gemm(num_tokens, hidden_size, khidden / 2, num_groups, fuse_config->gemm_sms)(
            o_vec.data_ptr(), o_scales_strided.data_ptr(),
            std::get<0>(y_fp8_2).data_ptr(), std::get<1>(y_fp8_2).data_ptr(),
            out_2.data_ptr(),
            packed_recv_count.data_ptr(),
            expected_m,
            current_stream,
            fuse_config->gemm_sms
        );
        auto [
            combine_x,
            event_,
            hook_
        ] = buffer->low_latency_combine(out_2.view(packed_recv_x.sizes()), topk_ids, topk_weights, packed_recv_src_info, packed_recv_layout_range, num_max_dispatch_tokens_per_rank, num_experts, fuse_config->ep_sms, true/*use_fp8*/, false/*zero_copy*/, false/*async_finish*/, false/*return_recv_hook*/, std::nullopt/*run stream*/, std::nullopt/*out: inplace tensor*/);
        if (enable_profile) cudaProfilerStop();
        cudaDeviceSynchronize();
    }

public:
    SequenceMoE(uint64_t num_experts, uint64_t num_max_dispatch_tokens_per_rank, uint64_t khidden, uint64_t hidden_size, uint64_t num_tokens, 
        uint64_t num_topk, uint64_t world_size, c10::intrusive_ptr<ProcessGroupNCCL>& global_pg, bool enable_random = true): 
        BaseMoE(num_experts, num_max_dispatch_tokens_per_rank, khidden, hidden_size, num_tokens, num_topk, world_size, global_pg) {
            if (enable_random) std::tie(hidden_states, topk_ids, topk_weights, x_fp8, y_fp8, out, o_vec, o_scales, o_scales_strided, silu_out, x_fp8_2, y_fp8_2, out_2) = initialize_random_inputs(num_tokens, num_topk, num_groups, num_experts, m_max, hidden_size, khidden);
            else std::tie(out, o_vec, o_scales, o_scales_strided, silu_out, out_2) = initialize_emptys(num_tokens, num_topk, num_groups, num_experts, m_max, hidden_size, khidden);
            get_deepep_low_latency_buffer(num_max_dispatch_tokens_per_rank, hidden_size, global_pg, num_groups, buffer,
                true/*use_cuda_graph*/, std::nullopt, true/*use_fp8*/, num_experts, num_tokens);
    }
};