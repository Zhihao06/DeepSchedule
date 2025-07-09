#pragma once

using namespace deep_ep;
using namespace internode;
using namespace c10d;

void get_dispatch_config(uint64_t num_ranks, uint64_t num_sms, std::shared_ptr<Config>& config) {
    static const std::unordered_map<uint64_t, std::shared_ptr<Config>> config_map = {
        {2,   std::make_shared<Config>(num_sms, 16, 256, 6, 128)},
        {4,   std::make_shared<Config>(num_sms, 16, 256, 6, 128)},
        {8,   std::make_shared<Config>(num_sms, 6, 256, 6, 128)},
        {16,  std::make_shared<Config>(num_sms, 16, 288, 20, 128)},
        {24,  std::make_shared<Config>(num_sms, 8, 288, 32, 128)},
        {32,  std::make_shared<Config>(num_sms, 8, 288, 32, 128)},
        {64,  std::make_shared<Config>(num_sms, 20, 288, 28, 128)},
        {128, std::make_shared<Config>(num_sms, 20, 560, 32, 128)},
        {144, std::make_shared<Config>(num_sms, 32, 720, 12, 128)},
        {160, std::make_shared<Config>(num_sms, 28, 720, 12, 128)}
    };

    auto it = config_map.find(num_ranks);
    if (it != config_map.end()) {
        config = it->second;
    } else {
        throw std::invalid_argument("Unsupported number of EP ranks: " + std::to_string(num_ranks));
    }
}

void get_combine_config(uint64_t num_ranks, uint64_t num_sms, std::shared_ptr<Config>& config) {
    static const std::unordered_map<uint64_t, std::shared_ptr<Config>> config_map = {
        {2,   std::make_shared<Config>(num_sms, 6, 256, 6, 128)},
        {4,   std::make_shared<Config>(num_sms, 6, 256, 6, 128)},
        {8,   std::make_shared<Config>(num_sms, 6, 256, 6, 128)},
        {16,  std::make_shared<Config>(num_sms, 2, 288, 28, 128)},
        {24,  std::make_shared<Config>(num_sms, 1, 288, 20, 128)},
        {32,  std::make_shared<Config>(num_sms, 1, 288, 20, 128)},
        {64,  std::make_shared<Config>(num_sms, 1, 288, 20, 128)},
        {128, std::make_shared<Config>(num_sms, 1, 560, 12, 128)},
        {144, std::make_shared<Config>(num_sms, 2, 720, 8, 128)},
        {160, std::make_shared<Config>(num_sms, 2, 720, 8, 128)}
    };

    auto it = config_map.find(num_ranks);
    if (it != config_map.end()) {
        config = it->second;
    } else {
        throw std::invalid_argument("Unsupported number of EP ranks: " + std::to_string(num_ranks));
    }
}

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

void get_deepep_normal_buffer(uint64_t num_max_dispatch_tokens_per_rank, uint64_t hidden, uint64_t param_bytes, c10::intrusive_ptr<ProcessGroupNCCL>& group, uint64_t num_groups, std::shared_ptr<Buffer>& buffer, int num_sms, std::optional<c10::cuda::CUDAStream> comm_stream) {

    // Initialize the CPP runtime
    auto group_size = group->getSize();
    auto rank = group->getRank();
    auto hidden_bytes = hidden_size * param_bytes;
    std::shared_ptr<Config> dispatch_config, combine_config;
    get_dispatch_config(group_size, num_sms, dispatch_config);
    get_combine_config(group_size, num_sms, combine_config);
    auto num_nvl_bytes = std::max({
        dispatch_config->get_nvl_buffer_size_hint(hidden_bytes, group_size),
        combine_config->get_nvl_buffer_size_hint(hidden_bytes, group_size),
        0
    });
    auto num_rdma_bytes = std::max({
        dispatch_config->get_rdma_buffer_size_hint(hidden_bytes, group_size),
        combine_config->get_rdma_buffer_size_hint(hidden_bytes, group_size),
        0
    });
    auto low_latency_mode = false;
    auto num_qps_per_rank = num_groups;
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
    if (buffer->get_num_rdma_ranks() > 1) {
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
        root_unique_id = nvshmem_unique_ids[buffer->get_root_rdma_rank(true)];
    }

    pybind11::bytearray root_unique_id_arr = {reinterpret_cast<const char*>(root_unique_id.data()), root_unique_id.size()};

    // Make CPP runtime available
    buffer->sync(device_ids, ipc_handles, root_unique_id_arr);
    assert(buffer->is_available());
}
