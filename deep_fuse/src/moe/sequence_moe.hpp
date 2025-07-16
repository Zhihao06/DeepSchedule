#pragma once

#include "base_moe.hpp"

using namespace deep_ep;
using namespace c10d;

class SequenceMoE : public BaseMoE {
private:
    torch::Tensor packed_recv_x;
    std::optional<torch::Tensor> packed_recv_x_scales;
    torch::Tensor packed_recv_count;
    torch::Tensor packed_recv_src_info;
    torch::Tensor packed_recv_layout_range;
    std::optional<EventHandle> event;
    std::optional<std::function<void()>> hook;
    torch::Tensor combine_x;
    std::optional<EventHandle> event_c;
    std::optional<std::function<void()>> hook_c;

protected:
    void _compute_op(c10::cuda::CUDAStream current_stream, std::shared_ptr<FUSEConfig>& fuse_config) {
        // FC1
        get_function_for_gemm(num_tokens, khidden, hidden_size, num_groups, fuse_config->gemm_sms)(
            packed_recv_x.data_ptr(), packed_recv_x_scales.value().data_ptr(),
            std::get<0>(y_fp8).data_ptr(), std::get<1>(y_fp8).data_ptr(),
            out.data_ptr(),
            packed_recv_count.data_ptr(),
            expected_m,
            current_stream,
            fuse_config->gemm_sms
        );

        // silu
        auto out_size = static_cast<int64_t>(num_groups * m_max);
        auto out_view = out.view({out_size, -1});
        fuse_silu_and_mul_masked(silu_out, o_vec, o_scales, out_view, packed_recv_count, m_max, current_stream);
        o_scales_strided = get_col_major_tma_aligned_tensor_wrapper(o_scales, current_stream);

        // FC2
        get_function_for_gemm(num_tokens, hidden_size, khidden / 2, num_groups, fuse_config->gemm_sms)(
            o_vec.data_ptr(), o_scales_strided.data_ptr(),
            std::get<0>(y_fp8_2).data_ptr(), std::get<1>(y_fp8_2).data_ptr(),
            out_2.data_ptr(),
            packed_recv_count.data_ptr(),
            expected_m,
            current_stream,
            fuse_config->gemm_sms
        );
    }

    void _dispatch_op_a(c10::cuda::CUDAStream current_stream, std::shared_ptr<FUSEConfig>& fuse_config) {
        std::tie(packed_recv_x, packed_recv_x_scales, packed_recv_count, packed_recv_src_info, packed_recv_layout_range, event, hook) = buffer->low_latency_dispatch(hidden_states, topk_ids, num_max_dispatch_tokens_per_rank, num_experts, fuse_config->ep_sms, true/*use_fp8*/, false/*async_finish*/, true/*return_recv_hook*/, current_stream);
    }

    void _dispatch_op_b(c10::cuda::CUDAStream current_stream, std::shared_ptr<FUSEConfig>& fuse_config) {
        if (hook.has_value()) hook.value()();
    }

    void _combine_op_a(c10::cuda::CUDAStream current_stream, std::shared_ptr<FUSEConfig>& fuse_config) {
        std::tie(combine_x, event_c, hook_c) = buffer->low_latency_combine(out_2.view(packed_recv_x.sizes()), topk_ids, topk_weights, packed_recv_src_info, packed_recv_layout_range, num_max_dispatch_tokens_per_rank, num_experts, fuse_config->ep_sms, COMBINE_FP8/*use_fp8*/, false/*zero_copy*/, false/*async_finish*/, true/*return_recv_hook*/, current_stream/*run stream*/, std::nullopt/*out: inplace tensor*/);
    }

    void _combine_op_b(c10::cuda::CUDAStream current_stream, std::shared_ptr<FUSEConfig>& fuse_config) {
        if (hook_c.has_value()) hook_c.value()();
    }

    void _moe_core(std::shared_ptr<FUSEConfig>& fuse_config, LaunchMode launch_mode, bool enable_profile) {
        c10::cuda::CUDAStream current_stream = at::cuda::getCurrentCUDAStream();
        _dispatch_op_a(current_stream, fuse_config);
        _dispatch_op_b(current_stream, fuse_config);
        _compute_op(current_stream, fuse_config);
        _combine_op_a(current_stream, fuse_config);
        _combine_op_b(current_stream, fuse_config);
    }

public:
    SequenceMoE(uint64_t num_experts, uint64_t num_max_dispatch_tokens_per_rank, uint64_t khidden, uint64_t hidden_size, uint64_t num_tokens, 
        uint64_t num_topk, uint64_t world_size, c10::intrusive_ptr<ProcessGroupNCCL>& global_pg, bool enable_random = true): 
        BaseMoE(num_experts, num_max_dispatch_tokens_per_rank, khidden, hidden_size, num_tokens, num_topk, world_size, global_pg, enable_random) {
            get_deepep_low_latency_buffer(num_max_dispatch_tokens_per_rank, hidden_size, global_pg, num_groups, buffer, std::nullopt);
    }

    void get_metadata(uint64_t num_tokens) {
        this->num_tokens = num_tokens;
        expected_m = std::max(1UL, (num_tokens * num_topk + num_experts - 1) / num_experts * 2);
        if (enable_random) {
            std::tie(hidden_states, topk_ids, topk_weights, x_fp8, y_fp8, out, o_vec, o_scales, o_scales_strided, silu_out, x_fp8_2, y_fp8_2, out_2) = initialize_random_inputs(num_tokens, num_topk, num_groups, num_experts, m_max, hidden_size, khidden);
        } else {
            std::tie(out, o_vec, o_scales, o_scales_strided, silu_out, out_2) = initialize_empty_intermediate(num_tokens, num_topk, num_groups, num_experts, m_max, hidden_size, khidden);
        }
    }

    void load_weights(const torch::Tensor& w13_weight_data, const torch::Tensor& w13_weight_scale, const torch::Tensor& w2_weight_data, const torch::Tensor& w2_weight_scale) {
        y_fp8 = std::make_tuple(w13_weight_data, w13_weight_scale);
        y_fp8_2 = std::make_tuple(w2_weight_data, w2_weight_scale);
    }

    void load_inputs(const torch::Tensor& hidden_states_in, const torch::Tensor& topk_ids_in, const torch::Tensor& topk_weights_in) {
        hidden_states = hidden_states_in;
        topk_ids = topk_ids_in;
        topk_weights = topk_weights_in;
    }

    torch::Tensor get_merged_output() {
        torch::Tensor final_output = combine_x;
        assert(final_output.size(0) == num_tokens);
        assert(final_output.size(1) == hidden_size);
        return final_output;
    }

    void launch(std::shared_ptr<FUSEConfig>& fuse_config) {
        _moe_core(fuse_config, LaunchMode::DEFAULT_LAUNCH, false);
    }

    std::tuple<torch::Tensor, std::optional<torch::Tensor>, torch::Tensor, uint64_t>
    low_latency_dispatch_interface(std::shared_ptr<FUSEConfig>& fuse_config) {
        c10::cuda::CUDAStream current_stream = at::cuda::getCurrentCUDAStream();
        _dispatch_op_a(current_stream, fuse_config);
        _dispatch_op_b(current_stream, fuse_config);
        return std::make_tuple(packed_recv_x, packed_recv_x_scales, packed_recv_count, expected_m);
    }

    torch::Tensor
    low_latency_combine_interface(const torch::Tensor& compute_result, std::shared_ptr<FUSEConfig>& fuse_config) {
        c10::cuda::CUDAStream current_stream = at::cuda::getCurrentCUDAStream();
        out_2 = compute_result;
        _combine_op_a(current_stream, fuse_config);
        _combine_op_b(current_stream, fuse_config);
        return combine_x;
    }
};