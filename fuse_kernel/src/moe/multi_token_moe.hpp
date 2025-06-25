#include "base_moe.hpp"
#include "../gemm_gen.hpp"

using namespace deep_ep;
using namespace c10d;

class MultiTokenMoE : public BaseMoE {
private:
    uint64_t num_splits;
    std::vector<uint64_t> num_split_tokens;
    std::vector<uint64_t> expected_ms;
    std::vector<torch::Tensor> packed_recv_x;
    std::vector<std::optional<torch::Tensor>> packed_recv_x_scales;
    std::vector<torch::Tensor> packed_recv_count;
    std::vector<torch::Tensor> packed_recv_src_info;
    std::vector<torch::Tensor> packed_recv_layout_range;
    std::vector<std::optional<EventHandle>> events;
    std::vector<std::optional<std::function<void()>>> hooks;
    std::vector<torch::Tensor> combine_x;
    std::vector<std::optional<EventHandle>> event_cs;
    std::vector<std::optional<std::function<void()>>> hook_cs;

    void _compute_op(c10::cuda::CUDAStream current_stream, std::shared_ptr<FUSEConfig>& fuse_config, int index) {
        // FC1
        get_function_for_gemm(num_split_tokens[index], khidden, hidden_size, num_groups, fuse_config->gemm_sms)(
            std::get<0>(x_fp8[index]).data_ptr(), std::get<1>(x_fp8[index]).data_ptr(),
            std::get<0>(y_fp8[index]).data_ptr(), std::get<1>(y_fp8[index]).data_ptr(),
            out[index].data_ptr(),
            packed_recv_count[index].data_ptr(),
            expected_ms[index],
            current_stream,
            fuse_config->gemm_sms
        );

        // silu
        auto out_size = static_cast<int64_t>(num_groups * m_max);
        auto out_view = out[index].view({out_size, -1});
        fuse_silu_and_mul_masked(silu_out[index], o_vec[index], o_scales[index], out_view, packed_recv_count[index], m_max, current_stream);

        // FC2
        get_function_for_gemm(num_split_tokens[index], hidden_size, khidden / 2, num_groups, fuse_config->gemm_sms)(
            o_vec[index].data_ptr(), o_scales_strided[index].data_ptr(),
            std::get<0>(y_fp8_2[index]).data_ptr(), std::get<1>(y_fp8_2[index]).data_ptr(),
            out_2[index].data_ptr(),
            packed_recv_count[index].data_ptr(),
            expected_ms[index],
            current_stream,
            fuse_config->gemm_sms
        );
    }

    void _compute_fuse_op(c10::cuda::CUDAStream current_stream, std::shared_ptr<FUSEConfig>& fuse_config, int index) {
        _combine_op_a(comm_stream, fuse_config, index-1);
        // FC1
        get_function_for_gemm(num_split_tokens[index], khidden, hidden_size, num_groups, fuse_config->gemm_sms)(
            std::get<0>(x_fp8[index]).data_ptr(), std::get<1>(x_fp8[index]).data_ptr(),
            std::get<0>(y_fp8[index]).data_ptr(), std::get<1>(y_fp8[index]).data_ptr(),
            out[index].data_ptr(),
            packed_recv_count[index].data_ptr(),
            expected_ms[index],
            compute_stream,
            fuse_config->gemm_sms
        );
        _dispatch_op_a(comm_stream, fuse_config, index+1);

        // silu
        auto out_size = static_cast<int64_t>(num_groups * m_max);
        auto out_view = out[index].view({out_size, -1});
        fuse_silu_and_mul_masked(silu_out[index], o_vec[index], o_scales[index], out_view, packed_recv_count[index], m_max, compute_stream);

        // FC2
        _dispatch_op_b(comm_stream, fuse_config, index+1);
        get_function_for_gemm(num_split_tokens[index], hidden_size, khidden / 2, num_groups, fuse_config->gemm_sms)(
            o_vec[index].data_ptr(), o_scales_strided[index].data_ptr(),
            std::get<0>(y_fp8_2[index]).data_ptr(), std::get<1>(y_fp8_2[index]).data_ptr(),
            out_2[index].data_ptr(),
            packed_recv_count[index].data_ptr(),
            expected_ms[index],
            compute_stream,
            fuse_config->gemm_sms
        );
        // _combine_op_b(comm_stream, fuse_config, index-1);
    }

    void _dispatch_op(c10::cuda::CUDAStream current_stream, std::shared_ptr<FUSEConfig>& fuse_config, int index) {
        std::tie(packed_recv_x[index], packed_recv_x_scales[index], packed_recv_count[index], packed_recv_src_info[index], packed_recv_layout_range[index], events[index], hooks[index]) = buffers[index]->low_latency_dispatch(hidden_states[index], topk_ids[index], num_max_dispatch_tokens_per_rank, num_experts, fuse_config->ep_sms, true/*use_fp8*/, false/*async_finish*/, false/*return_recv_hook*/, current_stream);
    }

    void _combine_op(c10::cuda::CUDAStream current_stream, std::shared_ptr<FUSEConfig>& fuse_config, int index) {
        std::tie(combine_x[index], event_cs[index], hook_cs[index]) = buffers[index]->low_latency_combine(out_2[index].view(packed_recv_x[index].sizes()), topk_ids[index], topk_weights[index], packed_recv_src_info[index], packed_recv_layout_range[index], num_max_dispatch_tokens_per_rank, num_experts, fuse_config->ep_sms, true/*use_fp8*/, false/*zero_copy*/, false/*async_finish*/, false/*return_recv_hook*/, current_stream/*run stream*/, std::nullopt/*out: inplace tensor*/);
    }

    void _dispatch_op_a(c10::cuda::CUDAStream current_stream, std::shared_ptr<FUSEConfig>& fuse_config, int index) {
        std::tie(packed_recv_x[index], packed_recv_x_scales[index], packed_recv_count[index], packed_recv_src_info[index], packed_recv_layout_range[index], events[index], hooks[index]) = buffers[index]->low_latency_dispatch(hidden_states[index], topk_ids[index], num_max_dispatch_tokens_per_rank, num_experts, fuse_config->ep_sms, true/*use_fp8*/, false/*async_finish*/, true/*return_recv_hook*/, current_stream);
    }

    void _dispatch_op_b(c10::cuda::CUDAStream current_stream, std::shared_ptr<FUSEConfig>& fuse_config, int index) {
        if (hooks[index].has_value()) hooks[index].value()();
    }

    void _combine_op_a(c10::cuda::CUDAStream current_stream, std::shared_ptr<FUSEConfig>& fuse_config, int index) {
        std::tie(combine_x[index], event_cs[index], hook_cs[index]) = buffers[index]->low_latency_combine(out_2[index].view(packed_recv_x[index].sizes()), topk_ids[index], topk_weights[index], packed_recv_src_info[index], packed_recv_layout_range[index], num_max_dispatch_tokens_per_rank, num_experts, fuse_config->ep_sms, true/*use_fp8*/, false/*zero_copy*/, false/*async_finish*/, true/*return_recv_hook*/, current_stream/*run stream*/, std::nullopt/*out: inplace tensor*/);
    }

    void _combine_op_b(c10::cuda::CUDAStream current_stream, std::shared_ptr<FUSEConfig>& fuse_config, int index) {
        if (hook_cs[index].has_value()) hook_cs[index].value()();
    }

    void _moe_sync(std::shared_ptr<FUSEConfig>& fuse_config) {
        c10::cuda::CUDAStream current_stream = at::cuda::getCurrentCUDAStream();
        global_pg->barrier()->wait();

        // Dispatch 0 issue
        _dispatch_op_a(current_stream, fuse_config, 0);
        _dispatch_op_b(current_stream, fuse_config, 0);
        _dispatch_op_a(current_stream, fuse_config, 1);
        _compute_op(current_stream, fuse_config, 0);
        _dispatch_op_b(current_stream, fuse_config, 1);

        // Index 1 to num_splits - 2
        for (int i = 0; i < num_splits - 2; i++) {
            _combine_op_a(current_stream, fuse_config, i);
            _dispatch_op_a(current_stream, fuse_config, i + 2);
            _compute_op(current_stream, fuse_config, i + 1);
            _combine_op_b(current_stream, fuse_config, i);
            _dispatch_op_b(current_stream, fuse_config, i + 2);
        }

        _combine_op_a(current_stream, fuse_config, num_splits - 2);
        _compute_op(current_stream, fuse_config, num_splits - 1);
        _combine_op_b(current_stream, fuse_config, num_splits - 2);
        _combine_op_a(current_stream, fuse_config, num_splits - 1);
        _combine_op_b(current_stream, fuse_config, num_splits - 1);
    }

    void _moe_sched(std::shared_ptr<FUSEConfig>& fuse_config) {
        c10::cuda::CUDAStream current_stream = at::cuda::getCurrentCUDAStream();
        global_pg->barrier()->wait();

        // Dispatch 0 issue
        _dispatch_op_a(comm_stream, fuse_config, 0);
        _dispatch_op_b(comm_stream, fuse_config, 0);
        stream_wait(compute_stream, comm_stream);
        _compute_op(compute_stream, fuse_config, 0);
        _dispatch_op_a(comm_stream, fuse_config, 1);
        _dispatch_op_b(comm_stream, fuse_config, 1);

        // Index 1 to num_splits - 2
        for (int i = 0; i < num_splits - 2; i++) {
            stream_wait(compute_stream, comm_stream);
            stream_wait(comm_stream, compute_stream);
            _compute_fuse_op(compute_stream, fuse_config, i+1);
        }

        stream_wait(compute_stream, comm_stream);
        stream_wait(comm_stream, compute_stream);
        _combine_op_a(comm_stream, fuse_config, num_splits-2);
        _compute_op(compute_stream, fuse_config, num_splits-1);
        stream_wait(comm_stream, compute_stream);
        _combine_op_a(comm_stream, fuse_config, num_splits-1);
        for (int i = 0; i < num_splits - 1; i++) {
            _combine_op_b(comm_stream, fuse_config, i);
        }
        _combine_op_b(comm_stream, fuse_config, num_splits-1);

        cudaDeviceSynchronize();
    }

protected:
    void _moe_core(std::shared_ptr<FUSEConfig>& fuse_config, bool enable_profile) {
        // _moe_sync(fuse_config);
        _moe_sched(fuse_config);
    }

public:
    std::vector<torch::Tensor> hidden_states, topk_ids, topk_weights;
    std::vector<std::tuple<torch::Tensor, torch::Tensor>> x_fp8, y_fp8;
    std::vector<torch::Tensor> o_vec, o_scales, o_scales_strided, silu_out;
    std::vector<std::tuple<torch::Tensor, torch::Tensor>> x_fp8_2, y_fp8_2;
    std::vector<torch::Tensor> out, out_2;

    std::vector<std::shared_ptr<Buffer>> buffers;
    c10::cuda::CUDAStream comm_stream, compute_stream;

    MultiTokenMoE(uint64_t num_experts, uint64_t num_max_dispatch_tokens_per_rank, uint64_t khidden, uint64_t hidden_size, uint64_t num_tokens, 
        uint64_t num_topk, uint64_t world_size, std::shared_ptr<ProcessGroupNCCL>& global_pg, uint64_t num_splits, std::vector<uint64_t> num_split_tokens): 
        BaseMoE(num_experts, num_max_dispatch_tokens_per_rank, khidden, hidden_size, num_tokens, num_topk, world_size, global_pg),
        num_splits(num_splits), comm_stream(at::cuda::getStreamFromPool(true)), compute_stream(at::cuda::getStreamFromPool(true)) {
            assert(num_splits > 1);
            if (num_split_tokens.size() == 0) { // average split
                assert(num_tokens % num_splits == 0);
                this->num_split_tokens = std::vector<uint64_t>(num_splits, num_tokens / num_splits);
            } else { // manual split
                assert(num_split_tokens.size() == num_splits);
                assert(std::accumulate(num_split_tokens.begin(), num_split_tokens.end(), 0) == num_tokens);
                this->num_split_tokens = num_split_tokens;
            }

            // get default variables
            hidden_states.resize(num_splits);
            topk_ids.resize(num_splits);
            topk_weights.resize(num_splits);
            x_fp8.resize(num_splits);
            y_fp8.resize(num_splits);
            o_vec.resize(num_splits);
            o_scales.resize(num_splits);
            o_scales_strided.resize(num_splits);
            silu_out.resize(num_splits);
            x_fp8_2.resize(num_splits);
            y_fp8_2.resize(num_splits);
            out.resize(num_splits);
            out_2.resize(num_splits);
            buffers.resize(num_splits);

            expected_ms.resize(num_splits);
            packed_recv_x.resize(num_splits);
            packed_recv_x_scales.resize(num_splits);
            packed_recv_count.resize(num_splits);
            packed_recv_src_info.resize(num_splits);
            packed_recv_layout_range.resize(num_splits);
            events.resize(num_splits);
            hooks.resize(num_splits);
            combine_x.resize(num_splits);
            event_cs.resize(num_splits);
            hook_cs.resize(num_splits);

            for (int i = 0; i < num_splits; i++) {
                expected_ms[i] = std::max(1UL, (this->num_split_tokens[i] * num_topk + num_experts - 1) / num_experts * 2);
                std::tie(hidden_states[i], topk_ids[i], topk_weights[i], x_fp8[i], y_fp8[i], out[i], o_vec[i], o_scales[i], o_scales_strided[i], silu_out[i], x_fp8_2[i], y_fp8_2[i], out_2[i]) = initialize_random_inputs(this->num_split_tokens[i], num_topk, num_groups, num_experts, m_max, hidden_size, khidden);
                get_deepep_low_latency_buffer(num_max_dispatch_tokens_per_rank, hidden_size, global_pg, num_groups, buffers[i],
                    true/*use_cuda_graph*/, comm_stream, true/*use_fp8*/, num_experts, this->num_split_tokens[i]);
            }
    }
};