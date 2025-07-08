#pragma once

#include "base_moe.hpp"

using namespace deep_ep;
using namespace c10d;

class TBOMoE : public BaseMoE {
protected:
    std::shared_ptr<Buffer> buffer_b;

    void _moe_core(std::shared_ptr<FUSEConfig>& fuse_config, LaunchMode launch_mode, bool enable_profile) {
        cudaStream_t current_stream = at::cuda::getCurrentCUDAStream();
        assert(buffer_b != nullptr and hidden_states_b.defined());
        global_pg->barrier()->wait();

        // Dispatch 0 issue
        auto [
            packed_recv_x, 
            packed_recv_x_scales, 
            packed_recv_count, 
            packed_recv_src_info, 
            packed_recv_layout_range, 
            event, 
            hook
        ] = buffer->low_latency_dispatch(hidden_states, topk_ids, num_max_dispatch_tokens_per_rank, num_experts, fuse_config->ep_sms, true/*use_fp8*/, false/*async_finish*/, true/*return_recv_hook*/);

        // Dispatch 0 recv
        if (hook.has_value()) hook.value()();

        // Dispatch 1 issue
        auto [
            packed_recv_x_b, 
            packed_recv_x_scales_b, 
            packed_recv_count_b, 
            packed_recv_src_info_b, 
            packed_recv_layout_range_b, 
            event_b, 
            hook_b
        ] = buffer_b->low_latency_dispatch(hidden_states_b, topk_ids_b, num_max_dispatch_tokens_per_rank, num_experts, fuse_config->ep_sms, true/*use_fp8*/, false/*async_finish*/, true/*return_recv_hook*/);

        // 0: FC1
        get_function_for_gemm(num_tokens, khidden, hidden_size, num_groups, fuse_config->gemm_sms)(
            std::get<0>(x_fp8).data_ptr(), std::get<1>(x_fp8).data_ptr(),
            std::get<0>(y_fp8).data_ptr(), std::get<1>(y_fp8).data_ptr(),
            out.data_ptr(),
            packed_recv_count.data_ptr(),
            expected_m,
            current_stream,
            fuse_config->gemm_sms
        );

        // 0: silu
        auto out_size = static_cast<int64_t>(num_groups * m_max);
        auto out_view = out.view({out_size, -1});
        fuse_silu_and_mul_masked(silu_out, o_vec, o_scales, out_view, packed_recv_count, m_max);

        // 0: FC2
        get_function_for_gemm(num_tokens, hidden_size, khidden / 2, num_groups, fuse_config->gemm_sms)(
            o_vec.data_ptr(), o_scales_strided.data_ptr(),
            std::get<0>(y_fp8_2).data_ptr(), std::get<1>(y_fp8_2).data_ptr(),
            out_2.data_ptr(),
            packed_recv_count.data_ptr(),
            expected_m,
            current_stream,
            fuse_config->gemm_sms
        );

        // Dispatch 1 recv
        if (hook_b.has_value()) hook_b.value()();

        // Combine 0 issue
        auto [
            combine_x,
            event_,
            hook_
        ] = buffer->low_latency_combine(out_2.view(packed_recv_x.sizes()), topk_ids, topk_weights, packed_recv_src_info, packed_recv_layout_range, num_max_dispatch_tokens_per_rank, num_experts, fuse_config->ep_sms, false/*use_fp8*/, false/*zero_copy*/, false/*async_finish*/, true/*return_recv_hook*/, std::nullopt/*run stream*/, std::nullopt/*out: inplace tensor*/);

        // 1: FC1
        get_function_for_gemm(num_tokens, khidden, hidden_size, num_groups, fuse_config->gemm_sms)(
            std::get<0>(x_fp8_b).data_ptr(), std::get<1>(x_fp8_b).data_ptr(),
            std::get<0>(y_fp8_b).data_ptr(), std::get<1>(y_fp8_b).data_ptr(),
            out_b.data_ptr(),
            packed_recv_count_b.data_ptr(),
            expected_m,
            current_stream,
            fuse_config->gemm_sms
        );

        // 1: silu
        auto out_size_b = static_cast<int64_t>(num_groups * m_max);
        auto out_view_b = out_b.view({out_size_b, -1});
        fuse_silu_and_mul_masked(silu_out_b, o_vec_b, o_scales_b, out_view_b, packed_recv_count_b, m_max);

        // 1: FC2
        get_function_for_gemm(num_tokens, hidden_size, khidden / 2, num_groups, fuse_config->gemm_sms)(
            o_vec_b.data_ptr(), o_scales_strided_b.data_ptr(),
            std::get<0>(y_fp8_2_b).data_ptr(), std::get<1>(y_fp8_2_b).data_ptr(),
            out_2_b.data_ptr(),
            packed_recv_count_b.data_ptr(),
            expected_m,
            current_stream,
            fuse_config->gemm_sms
        );

        // Combine 0 recv
        if (hook_.has_value()) hook_.value()();

        // Combine 1 issue
        auto [
            combine_x_b,
            event_b_,
            hook_b_
        ] = buffer_b->low_latency_combine(out_2_b.view(packed_recv_x_b.sizes()), topk_ids_b, topk_weights_b, packed_recv_src_info_b, packed_recv_layout_range_b, num_max_dispatch_tokens_per_rank, num_experts, fuse_config->ep_sms, false/*use_fp8*/, false/*zero_copy*/, false/*async_finish*/, true/*return_recv_hook*/, std::nullopt/*run stream*/, std::nullopt/*out: inplace tensor*/);

        // Combine 1 recv
        if (hook_b_.has_value()) hook_b_.value()();

        cudaDeviceSynchronize();
    }

public:
    torch::Tensor hidden_states_b, topk_ids_b, topk_weights_b;
    std::tuple<torch::Tensor, torch::Tensor> x_fp8_b, y_fp8_b;
    torch::Tensor o_vec_b, o_scales_b, o_scales_strided_b, silu_out_b;
    std::tuple<torch::Tensor, torch::Tensor> x_fp8_2_b, y_fp8_2_b;
    torch::Tensor out_b, out_2_b;

    TBOMoE(uint64_t num_experts, uint64_t num_max_dispatch_tokens_per_rank, uint64_t khidden, uint64_t hidden_size, uint64_t num_tokens, 
        uint64_t num_topk, uint64_t world_size, c10::intrusive_ptr<ProcessGroupNCCL>& global_pg, bool enable_random = true): 
        BaseMoE(num_experts, num_max_dispatch_tokens_per_rank, khidden, hidden_size, num_tokens, num_topk, world_size, global_pg, enable_random) {
            std::tie(hidden_states, topk_ids, topk_weights, x_fp8, y_fp8, out, o_vec, o_scales, o_scales_strided, silu_out, x_fp8_2, y_fp8_2, out_2) = initialize_random_inputs(num_tokens, num_topk, num_groups, num_experts, m_max, hidden_size, khidden);
            std::tie(hidden_states_b, topk_ids_b, topk_weights_b, x_fp8_b, y_fp8_b, out_b, o_vec_b, o_scales_b, o_scales_strided_b, silu_out_b, x_fp8_2_b, y_fp8_2_b, out_2_b) = initialize_random_inputs(num_tokens, num_topk, num_groups, num_experts, m_max, hidden_size, khidden);
            get_deepep_low_latency_buffer(num_max_dispatch_tokens_per_rank, hidden_size, global_pg, num_groups, buffer, std::nullopt);
            get_deepep_low_latency_buffer(num_max_dispatch_tokens_per_rank, hidden_size, global_pg, num_groups, buffer_b, std::nullopt);
    }
};