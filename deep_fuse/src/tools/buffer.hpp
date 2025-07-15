#pragma once

using namespace deep_ep;
using namespace internode;
using namespace c10d;

void get_deepep_low_latency_buffer(uint64_t num_max_dispatch_tokens_per_rank, uint64_t hidden, c10::intrusive_ptr<ProcessGroupNCCL>& group, uint64_t num_groups, std::shared_ptr<Buffer>& buffer, std::optional<c10::cuda::CUDAStream> comm_stream) {

    // Initialize the CPP runtime
    auto group_size = group->getSize();
    auto rank = group->getRank();
    DEBUG_CODE(print_object(group_size, "group_size"));
    DEBUG_CODE(print_object(rank, "rank"));
    auto num_nvl_bytes = 0;
    auto low_latency_mode = true;
    auto num_qps_per_rank = num_groups;
    auto num_rdma_bytes = get_low_latency_rdma_size_hint(num_max_dispatch_tokens_per_rank, hidden, group_size, num_groups * group_size);
    buffer = std::make_shared<Buffer>(rank, group_size, num_nvl_bytes, num_rdma_bytes, low_latency_mode, comm_stream);
    
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
    setenv("NVSHMEM_QP_DEPTH", "4096", 1); // TODO: Tuning for performance
    setenv("NVSHMEM_CUMEM_GRANULARITY", "536870912", 1);  // 2^29 = 536,870,912
    setenv("RANK", std::to_string(rank).c_str(), 1);
    setenv("GROUP_SIZE", std::to_string(group_size).c_str(), 1);

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
