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
            packed_recv_x[index].data_ptr(), packed_recv_x_scales[index].value().data_ptr(),
            std::get<0>(y_fp8).data_ptr(), std::get<1>(y_fp8).data_ptr(),
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
        o_scales_strided[index] = get_col_major_tma_aligned_tensor_wrapper(o_scales[index], current_stream);

        // FC2
        get_function_for_gemm(num_split_tokens[index], hidden_size, khidden / 2, num_groups, fuse_config->gemm_sms)(
            o_vec[index].data_ptr(), o_scales_strided[index].data_ptr(),
            std::get<0>(y_fp8_2).data_ptr(), std::get<1>(y_fp8_2).data_ptr(),
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
            packed_recv_x[index].data_ptr(), packed_recv_x_scales[index].value().data_ptr(),
            std::get<0>(y_fp8).data_ptr(), std::get<1>(y_fp8).data_ptr(),
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
        o_scales_strided[index] = get_col_major_tma_aligned_tensor_wrapper(o_scales[index], compute_stream);

        // FC2
        _dispatch_op_b(comm_stream, fuse_config, index+1);
        get_function_for_gemm(num_split_tokens[index], hidden_size, khidden / 2, num_groups, fuse_config->gemm_sms)(
            o_vec[index].data_ptr(), o_scales_strided[index].data_ptr(),
            std::get<0>(y_fp8_2).data_ptr(), std::get<1>(y_fp8_2).data_ptr(),
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
        DEBUG_VAR("[ _dispatch_op_a ] ", index, ": ", hidden_states[index].sizes(), " ", topk_ids[index].sizes());
        std::tie(packed_recv_x[index], packed_recv_x_scales[index], packed_recv_count[index], packed_recv_src_info[index], packed_recv_layout_range[index], events[index], hooks[index]) = buffers[index]->low_latency_dispatch(hidden_states[index], topk_ids[index], num_max_dispatch_tokens_per_rank, num_experts, fuse_config->ep_sms, true/*use_fp8*/, false/*async_finish*/, true/*return_recv_hook*/, current_stream);
    }

    void _dispatch_op_b(c10::cuda::CUDAStream current_stream, std::shared_ptr<FUSEConfig>& fuse_config, int index) {
        if (hooks[index].has_value()) hooks[index].value()();
        DEBUG_VAR("[ _dispatch_op_b ] ", index, ": ", packed_recv_x[index].sizes(), " ", packed_recv_x_scales[index].value().sizes(), " ", packed_recv_count[index].sizes(), " ", packed_recv_src_info[index].sizes(), " ", packed_recv_layout_range[index].sizes());
    }

    void _combine_op_a(c10::cuda::CUDAStream current_stream, std::shared_ptr<FUSEConfig>& fuse_config, int index) {
        DEBUG_VAR("[ _combine_op_a ] ", index, ": ", out_2[index].view(packed_recv_x[index].sizes()).sizes(), " ", topk_ids[index].sizes(), " ", topk_weights[index].sizes(), " ", packed_recv_src_info[index].sizes(), " ", packed_recv_layout_range[index].sizes());
        std::tie(combine_x[index], event_cs[index], hook_cs[index]) = buffers[index]->low_latency_combine(out_2[index].view(packed_recv_x[index].sizes()), topk_ids[index], topk_weights[index], packed_recv_src_info[index], packed_recv_layout_range[index], num_max_dispatch_tokens_per_rank, num_experts, fuse_config->ep_sms, true/*use_fp8*/, false/*zero_copy*/, false/*async_finish*/, true/*return_recv_hook*/, current_stream/*run stream*/, std::nullopt/*out: inplace tensor*/);
    }

    void _combine_op_b(c10::cuda::CUDAStream current_stream, std::shared_ptr<FUSEConfig>& fuse_config, int index) {
        if (hook_cs[index].has_value()) hook_cs[index].value()();
        DEBUG_VAR("[ _combine_op_b ] ", index, ": ", combine_x[index].sizes(), " ");
    }

    void _moe_sync(std::shared_ptr<FUSEConfig>& fuse_config) {
        c10::cuda::CUDAStream current_stream = at::cuda::getCurrentCUDAStream();
        // global_pg->barrier()->wait();

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
        // global_pg->barrier()->wait();
        stream_wait(compute_stream, current_stream);
        stream_wait(comm_stream, current_stream);

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

        stream_wait(current_stream, comm_stream);
        stream_wait(current_stream, compute_stream);
    }

protected:
    void _moe_core(std::shared_ptr<FUSEConfig>& fuse_config, LaunchMode launch_mode, bool enable_profile) {
        assert(hidden_states.size() > 0 and topk_ids.size() > 0 and std::get<0>(y_fp8).defined() and std::get<0>(y_fp8_2).defined());
        if (launch_mode == LaunchMode::SCHED_LAUNCH) _moe_sched(fuse_config);
        else _moe_sync(fuse_config);
    }

public:
    std::vector<torch::Tensor> hidden_states, topk_ids, topk_weights;
    std::vector<std::tuple<torch::Tensor, torch::Tensor>> x_fp8;
    std::tuple<torch::Tensor, torch::Tensor> y_fp8;
    std::vector<torch::Tensor> o_vec, o_scales, o_scales_strided, silu_out;
    std::vector<std::tuple<torch::Tensor, torch::Tensor>> x_fp8_2;
    std::tuple<torch::Tensor, torch::Tensor> y_fp8_2;
    std::vector<torch::Tensor> out, out_2;

    std::vector<std::shared_ptr<Buffer>> buffers;
    c10::cuda::CUDAStream comm_stream, compute_stream;
    torch::Tensor final_output;

    MultiTokenMoE(uint64_t num_experts, uint64_t num_max_dispatch_tokens_per_rank, uint64_t khidden, uint64_t hidden_size, uint64_t num_tokens, 
        uint64_t num_topk, uint64_t world_size, c10::intrusive_ptr<ProcessGroupNCCL>& global_pg, bool enable_random = true, uint64_t num_splits = 2): 
        BaseMoE(num_experts, num_max_dispatch_tokens_per_rank, khidden, hidden_size, num_tokens, num_topk, world_size, global_pg, enable_random),
        num_splits(num_splits), comm_stream(at::cuda::getStreamFromPool(true)), compute_stream(at::cuda::getStreamFromPool(true)) {
            assert(num_splits > 1);

            // get default variables
            hidden_states.resize(num_splits);
            topk_ids.resize(num_splits);
            topk_weights.resize(num_splits);
            x_fp8.resize(num_splits);
            // y_fp8.resize(num_splits);
            o_vec.resize(num_splits);
            o_scales.resize(num_splits);
            o_scales_strided.resize(num_splits);
            silu_out.resize(num_splits);
            x_fp8_2.resize(num_splits);
            // y_fp8_2.resize(num_splits);
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
                get_deepep_low_latency_buffer(num_max_dispatch_tokens_per_rank, hidden_size, global_pg, num_groups, buffers[i], comm_stream);
            }
    }

    void get_split_metadata(uint64_t num_tokens, std::vector<uint64_t> num_split_tokens = {}) {
        this->num_tokens = num_tokens;
        if (num_split_tokens.size() == 0) { // average split
            assert(num_tokens % num_splits == 0);
            this->num_split_tokens = std::vector<uint64_t>(num_splits, num_tokens / num_splits);
        } else { // manual split
            assert(num_split_tokens.size() == num_splits);
            assert(std::accumulate(num_split_tokens.begin(), num_split_tokens.end(), 0) == num_tokens);
            this->num_split_tokens = num_split_tokens;
        }
        for (int i = 0; i < num_splits; i++) {
            expected_ms[i] = std::max(1UL, (this->num_split_tokens[i] * num_topk + num_experts - 1) / num_experts * 2);
            if (enable_random) {
                std::tie(hidden_states[i], topk_ids[i], topk_weights[i], x_fp8[i], y_fp8, out[i], o_vec[i], o_scales[i], o_scales_strided[i], silu_out[i], x_fp8_2[i], y_fp8_2, out_2[i]) = initialize_random_inputs(this->num_split_tokens[i], num_topk, num_groups, num_experts, m_max, hidden_size, khidden);
            } else {
                std::tie(out[i], o_vec[i], o_scales[i], o_scales_strided[i], silu_out[i], out_2[i]) = initialize_empty_intermediate(this->num_split_tokens[i], num_topk, num_groups, num_experts, m_max, hidden_size, khidden);
            }
        }
        final_output = torch::empty({num_tokens, hidden_size}, dtype(torch::kBFloat16).device(torch::kCUDA));
    }

    void load_weights(const torch::Tensor& w13_weight_data, const torch::Tensor& w13_weight_scale, const torch::Tensor& w2_weight_data, const torch::Tensor& w2_weight_scale) {
        y_fp8 = std::make_tuple(w13_weight_data, w13_weight_scale);
        y_fp8_2 = std::make_tuple(w2_weight_data, w2_weight_scale);
    }

    void load_inputs_and_split(const torch::Tensor& hidden_states_in, const torch::Tensor& topk_ids_in, const torch::Tensor& topk_weights_in) {
        assert(this->num_split_tokens.size() > 0);
        hidden_states = custom_split(hidden_states_in, this->num_split_tokens, 0);
        topk_ids = custom_split(topk_ids_in, this->num_split_tokens, 0);
        topk_weights = custom_split(topk_weights_in, this->num_split_tokens, 0);
        combine_x = custom_split(final_output, this->num_split_tokens, 0);
    }

    torch::Tensor get_merged_output() {
        return this->final_output;
    }

    void launch(LaunchMode launch_mode, std::shared_ptr<FUSEConfig>& fuse_config) {
        assert(this->num_split_tokens.size() > 0);
        if (launch_mode == LaunchMode::SYNC_LAUNCH) _moe_sync(fuse_config);
        else if (launch_mode == LaunchMode::SCHED_LAUNCH) _moe_sched(fuse_config);
        else throw std::runtime_error("Invalid launch mode");
    }
};