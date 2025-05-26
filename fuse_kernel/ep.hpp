#pragma once

#include <torch/torch.h>
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/TCPStore.hpp>
#include <pybind11/functional.h>
#include <cassert>

#include "deep_ep.hpp"
#include "config.hpp"
#include "tools/data.hpp"
#include "tools/gemm.hpp"
#include "tools/utils.hpp"
#include "tools/debug.hpp"

using namespace deep_ep;
using namespace c10d;
using namespace internode;

std::shared_ptr<ProcessGroupNCCL> global_pg;

void init_distributed(int rank, int world_size) {
    std::string host = "localhost";
    uint16_t port = 29500;

    c10d::TCPStoreOptions opts;
    opts.isServer = (rank == 0);
    opts.numWorkers = world_size;

    c10::intrusive_ptr<Store> store = c10::make_intrusive<TCPStore>("localhost", opts);
    global_pg = std::make_shared<c10d::ProcessGroupNCCL>(store, rank, world_size);
}

void destroy_distributed() {
    cudaDeviceSynchronize();
    global_pg->barrier()->wait();
    global_pg->shutdown();
}

void get_deepep_low_latency_buffer(uint64_t num_max_dispatch_tokens_per_rank, uint64_t hidden, std::shared_ptr<ProcessGroupNCCL>& group, uint64_t num_groups, std::shared_ptr<Buffer>& buffer) {

    // Initialize the CPP runtime
    auto group_size = group->getSize();
    auto rank = group->getRank();
    DEBUG_CODE(print_object(group_size, "group_size"));
    DEBUG_CODE(print_object(rank, "rank"));
    auto num_nvl_bytes = 0;
    auto low_latency_mode = true;
    auto num_qps_per_rank = num_groups;
    auto num_rdma_bytes = get_low_latency_rdma_size_hint(num_max_dispatch_tokens_per_rank, hidden, group_size, num_groups * group_size);
    buffer = std::make_shared<Buffer>(rank, group_size, num_nvl_bytes, num_rdma_bytes, low_latency_mode);
    
    // Synchronize device IDs
    int local_device_id = buffer->get_local_device_id();
    std::vector<int> device_ids(group_size);
    all_gather_wrap(device_ids, local_device_id, group);

    // Synchronize IPC handles
    std::optional<pybind11::bytearray> local_ipc_handle = buffer->get_local_ipc_handle();
    std::vector<std::optional<pybind11::bytearray>> ipc_handles(group_size);
    all_gather_wrap(ipc_handles, local_ipc_handle, group);

    // Synchronize NVSHMEM unique IDs
    std::vector<uint8_t> root_unique_id;
    setenv("NVSHMEM_DISABLE_P2P", "1", 1);
    setenv("NVSHMEM_IB_ENABLE_IBGDA", "1", 1);
    setenv("NVSHMEM_IBGDA_NIC_HANDLER", "gpu", 1);
    setenv(("NVSHMEM_IBGDA_NUM_RC_PER_PE"), std::to_string(num_qps_per_rank).c_str(), 1);
    setenv("NVSHMEM_QP_DEPTH", "1024", 1);
    setenv("NVSHMEM_CUMEM_GRANULARITY", "536870912", 1);  // 2^29 = 536,870,912

    std::vector<std::vector<uint8_t>> nvshmem_unique_ids(group_size);

    if ((low_latency_mode && rank == 0) ||
        (!low_latency_mode && buffer->get_rdma_rank() == 0)) {
        root_unique_id = internode::get_unique_id();
    }

    all_gather_wrap(nvshmem_unique_ids, root_unique_id, group);
    root_unique_id = nvshmem_unique_ids[0];
    pybind11::bytearray root_unique_id_arr = {reinterpret_cast<const char*>(root_unique_id.data()), root_unique_id.size()};

    // Make CPP runtime available
    buffer->sync(device_ids, ipc_handles, root_unique_id_arr);
    assert(buffer->is_available());
}

void ep_moe_core_(uint64_t num_experts, uint64_t num_max_dispatch_tokens_per_rank, uint64_t khidden, uint64_t hidden_size, uint64_t num_tokens,
    uint64_t num_topk, uint64_t world_size, uint64_t num_groups, uint64_t expected_m, uint64_t m_max, cudaStream_t current_stream, 
    std::shared_ptr<Buffer>& buffer, 
    torch::Tensor hidden_states,
    torch::Tensor topk_ids,
    torch::Tensor topk_weights,
    std::tuple<torch::Tensor, torch::Tensor> x_fp8,
    std::tuple<torch::Tensor, torch::Tensor> y_fp8,
    torch::Tensor out,
    std::tuple<torch::Tensor, torch::Tensor> x_fp8_2,
    std::tuple<torch::Tensor, torch::Tensor> y_fp8_2,
    torch::Tensor out_2) {
        auto [
            packed_recv_x, 
            packed_recv_x_scales, 
            packed_recv_count, 
            packed_recv_src_info, 
            packed_recv_layout_range, 
            event, 
            hook
        ] = buffer->low_latency_dispatch(hidden_states, topk_ids, num_max_dispatch_tokens_per_rank, num_experts, true/*use_fp8*/, false/*async_finish*/, false/*return_recv_hook*/);
        auto handle = std::make_tuple(packed_recv_src_info, packed_recv_layout_range, num_max_dispatch_tokens_per_rank, hidden_states.size(1), num_experts);
        launch_gemm(
            std::get<0>(x_fp8).data_ptr(), std::get<1>(x_fp8).data_ptr(),
            std::get<0>(y_fp8).data_ptr(), std::get<1>(y_fp8).data_ptr(),
            out.data_ptr(),
            packed_recv_count.data_ptr(),
            expected_m,
            current_stream
        );
        launch_gemm(
            std::get<0>(x_fp8_2).data_ptr(), std::get<1>(x_fp8_2).data_ptr(),
            std::get<0>(y_fp8_2).data_ptr(), std::get<1>(y_fp8_2).data_ptr(),
            out_2.data_ptr(),
            packed_recv_count.data_ptr(),
            expected_m,
            current_stream
        );
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
        ] = buffer->low_latency_combine(out_2.view(packed_recv_x.sizes()), topk_ids, topk_weights, src_info, layout_range, num_max_dispatch_tokens_per_rank, num_experts, false/*zero_copy*/, false/*async_finish*/, false/*return_recv_hook*/, std::nullopt/*out: inplace tensor*/);
}

void ep_moe(uint64_t num_experts, uint64_t num_max_dispatch_tokens_per_rank, uint64_t khidden, uint64_t hidden_size, uint64_t num_tokens, uint64_t num_topk, uint64_t world_size) {
    auto num_groups = num_experts / world_size;
    auto expected_m = std::max(1UL, (num_tokens * num_topk + num_experts - 1) / num_experts * 2);
    auto m_max = num_max_dispatch_tokens_per_rank * world_size;

    cudaStream_t current_stream;
    CUDA_CHECK(cudaStreamCreate(&current_stream));
    global_pg->barrier()->wait();
    std::shared_ptr<Buffer> buffer;
    get_deepep_low_latency_buffer(num_max_dispatch_tokens_per_rank, hidden_size, global_pg, num_groups, buffer);
    auto [
        hidden_states,
        topk_ids,
        topk_weights,
        x_fp8,
        y_fp8,
        out,
        x_fp8_2,
        y_fp8_2,
        out_2
    ] = initialize_random_inputs(num_tokens, num_topk, num_groups, num_experts, m_max, hidden_size, khidden);
    auto [
        packed_recv_x, 
        packed_recv_x_scales, 
        packed_recv_count, 
        packed_recv_src_info, 
        packed_recv_layout_range, 
        event, 
        hook
    ] = buffer->low_latency_dispatch(hidden_states, topk_ids, num_max_dispatch_tokens_per_rank, num_experts, true/*use_fp8*/, false/*async_finish*/, false/*return_recv_hook*/);
    auto handle = std::make_tuple(packed_recv_src_info, packed_recv_layout_range, num_max_dispatch_tokens_per_rank, hidden_states.size(1), num_experts);
    launch_gemm(
        std::get<0>(x_fp8).data_ptr(), std::get<1>(x_fp8).data_ptr(),
        std::get<0>(y_fp8).data_ptr(), std::get<1>(y_fp8).data_ptr(),
        out.data_ptr(),
        packed_recv_count.data_ptr(),
        expected_m,
        current_stream
    );
    launch_gemm(
        std::get<0>(x_fp8_2).data_ptr(), std::get<1>(x_fp8_2).data_ptr(),
        std::get<0>(y_fp8_2).data_ptr(), std::get<1>(y_fp8_2).data_ptr(),
        out_2.data_ptr(),
        packed_recv_count.data_ptr(),
        expected_m,
        current_stream
    );
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
    ] = buffer->low_latency_combine(out_2.view(packed_recv_x.sizes()), topk_ids, topk_weights, src_info, layout_range, num_max_dispatch_tokens_per_rank, num_experts, false/*zero_copy*/, false/*async_finish*/, false/*return_recv_hook*/, std::nullopt/*out: inplace tensor*/);
}