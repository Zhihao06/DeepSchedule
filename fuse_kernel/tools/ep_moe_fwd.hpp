#pragma once

using namespace deep_ep;
using namespace c10d;

class EPMoE {
private:
    uint64_t num_tokens, num_topk, num_groups, num_experts, m_max, hidden_size, khidden, num_max_dispatch_tokens_per_rank, expected_m, world_size;
    std::shared_ptr<ProcessGroupNCCL> global_pg;
    std::shared_ptr<Buffer> buffer;

    // For TBO
    std::shared_ptr<Buffer> buffer_b;

    void _ep_moe_core(std::shared_ptr<FUSEConfig>& fuse_config, bool enable_profile) {
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
        auto handle = std::make_tuple(packed_recv_src_info, packed_recv_layout_range, num_max_dispatch_tokens_per_rank, hidden_states.size(1), num_experts);
        cudaDeviceSynchronize();
        get_function_for_gemm(num_tokens, khidden, hidden_size, num_groups, fuse_config->gemm_sms)(
            std::get<0>(x_fp8).data_ptr(), std::get<1>(x_fp8).data_ptr(),
            std::get<0>(y_fp8).data_ptr(), std::get<1>(y_fp8).data_ptr(),
            out.data_ptr(),
            packed_recv_count.data_ptr(),
            expected_m,
            current_stream,
            fuse_config->gemm_sms
        );
        get_function_for_gemm(num_tokens, hidden_size, khidden / 2, num_groups, fuse_config->gemm_sms)(
            std::get<0>(x_fp8_2).data_ptr(), std::get<1>(x_fp8_2).data_ptr(),
            std::get<0>(y_fp8_2).data_ptr(), std::get<1>(y_fp8_2).data_ptr(),
            out_2.data_ptr(),
            packed_recv_count.data_ptr(),
            expected_m,
            current_stream,
            fuse_config->gemm_sms
        );
        cudaDeviceSynchronize();
        auto [
            src_info, 
            layout_range, 
            num_max_dispatch_tokens_per_rank_, 
            hidden, 
            num_experts_
        ] = handle;
        auto [
            combine_x,
            event_,
            hook_
        ] = buffer->low_latency_combine(out_2.view(packed_recv_x.sizes()), topk_ids, topk_weights, src_info, layout_range, num_max_dispatch_tokens_per_rank, num_experts, fuse_config->ep_sms, false/*zero_copy*/, false/*async_finish*/, false/*return_recv_hook*/, std::nullopt/*out: inplace tensor*/);
        if (enable_profile) cudaProfilerStop();
        cudaDeviceSynchronize();
    }

    void _ep_moe_overlap(std::shared_ptr<FUSEConfig>& fuse_config, bool enable_profile) {
        cudaStream_t current_stream = at::cuda::getCurrentCUDAStream();
        CUDA_CHECK(cudaStreamCreateWithFlags(&current_stream, cudaStreamNonBlocking));
        if (enable_profile) cudaProfilerStart();
        global_pg->barrier()->wait();
        torch::Tensor packed_recv_count_2 = torch::bincount(topk_ids.view(num_tokens * num_topk), {}, num_experts).to(torch::kCUDA) * world_size;
        packed_recv_count_2 = packed_recv_count_2.index({torch::indexing::Slice(global_pg->getRank() * num_groups, (global_pg->getRank() + 1) * num_groups)});
        cudaDeviceSynchronize();
        
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
            packed_recv_count_2.data_ptr(),
            expected_m,
            current_stream,
            fuse_config->gemm_sms
        );
        cudaDeviceSynchronize();

        auto handle = std::make_tuple(packed_recv_src_info, packed_recv_layout_range, num_max_dispatch_tokens_per_rank, hidden_states.size(1), num_experts);
        torch::Tensor out_2_comm = torch::randn(packed_recv_x.sizes(), dtype(torch::kBFloat16).device(torch::kCUDA));
        auto [
            src_info, 
            layout_range, 
            num_max_dispatch_tokens_per_rank_, 
            hidden, 
            num_experts_
        ] = handle;
        cudaDeviceSynchronize();

        get_function_for_gemm(num_tokens, hidden_size, khidden / 2, num_groups, fuse_config->gemm_sms)(
            std::get<0>(x_fp8_2).data_ptr(), std::get<1>(x_fp8_2).data_ptr(),
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
        ] = buffer->low_latency_combine(out_2_comm, topk_ids, topk_weights, src_info, layout_range, num_max_dispatch_tokens_per_rank, num_experts, fuse_config->ep_sms, false/*zero_copy*/, true/*async_finish*/, false/*return_recv_hook*/, std::nullopt/*out: inplace tensor*/);

        if (enable_profile) cudaProfilerStop();
        cudaDeviceSynchronize();
    }


public:
    torch::Tensor hidden_states, topk_ids, topk_weights;
    std::tuple<torch::Tensor, torch::Tensor> x_fp8, y_fp8;
    std::tuple<torch::Tensor, torch::Tensor> x_fp8_2, y_fp8_2;
    torch::Tensor out, out_2;

    // For TBO
    torch::Tensor hidden_states_b, topk_ids_b, topk_weights_b;
    std::tuple<torch::Tensor, torch::Tensor> x_fp8_b, y_fp8_b;
    std::tuple<torch::Tensor, torch::Tensor> x_fp8_2_b, y_fp8_2_b;
    torch::Tensor out_b, out_2_b;

    EPMoE(uint64_t num_experts, uint64_t num_max_dispatch_tokens_per_rank, uint64_t khidden, uint64_t hidden_size, uint64_t num_tokens, 
        uint64_t num_topk, uint64_t world_size, std::shared_ptr<ProcessGroupNCCL>& global_pg, std::shared_ptr<Buffer>& buffer, std::shared_ptr<Buffer>& buffer_b,
        bool enable_tbo):
        num_experts(num_experts), num_max_dispatch_tokens_per_rank(num_max_dispatch_tokens_per_rank), 
        khidden(khidden), hidden_size(hidden_size), num_tokens(num_tokens), num_topk(num_topk), 
        world_size(world_size), global_pg(global_pg), buffer(buffer), buffer_b(buffer_b) {
        num_groups = num_experts / world_size;
        expected_m = std::max(1UL, (num_tokens * num_topk + num_experts - 1) / num_experts * 2);
        m_max = num_max_dispatch_tokens_per_rank * world_size;
        std::tie(hidden_states, topk_ids, topk_weights, x_fp8, y_fp8, out, x_fp8_2, y_fp8_2, out_2) = initialize_random_inputs(num_tokens, num_topk, num_groups, num_experts, m_max, hidden_size, khidden);
        if (enable_tbo) { 
            std::tie(hidden_states_b, topk_ids_b, topk_weights_b, x_fp8_b, y_fp8_b, out_b, x_fp8_2_b, y_fp8_2_b, out_2_b) = initialize_random_inputs(num_tokens, num_topk, num_groups, num_experts, m_max, hidden_size, khidden);
        }
    }

    ~EPMoE() {
    }

    void ep_moe_core(std::vector<int> ep_sms, int repeat_times, bool enable_profile) {
        for (auto ep_sm: ep_sms) {
            std::shared_ptr<FUSEConfig> fuse_config = std::make_shared<FUSEConfig>(78-ep_sm, ep_sm);
            for (int i = 0; i < repeat_times; i++) {
                _ep_moe_core(fuse_config, enable_profile);
            }
        }
    }

    void ep_moe_overlap(std::vector<int> ep_sms, int repeat_times, bool enable_profile) { 
        for (auto ep_sm: ep_sms) {
            std::shared_ptr<FUSEConfig> fuse_config = std::make_shared<FUSEConfig>(78-ep_sm, ep_sm);
            for (int i = 0; i < repeat_times; i++) {
                _ep_moe_overlap(fuse_config, enable_profile);
            }
        }
    }
};