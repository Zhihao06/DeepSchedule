#include <cuda.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDADataType.h>
#include <torch/torch.h>
#include <torch/python.h>
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/TCPStore.hpp>
#include <pybind11/functional.h>
#include <cassert>

#include "deep_ep.hpp"
#include "config.hpp"
#include "tools/utils.hpp"
#include "tools/data.hpp"
#include "tools/debug.hpp"
#include "tools/config.hpp"
#include "gemm_gen.hpp"

#include "moe/base_moe.hpp"
#include "moe/sequence_moe.hpp"
#include "moe/overlap_moe.hpp"
#include "moe/tbo_moe.hpp"
#include "moe/multi_token_moe.hpp"

#include "deep_fuse.hpp"

using namespace deep_ep;
using namespace c10d;

namespace deep_fuse { 

Tool::Tool(uint64_t num_experts, uint64_t num_max_dispatch_tokens_per_rank, uint64_t khidden, uint64_t hidden_size, uint64_t num_tokens, 
    uint64_t num_topk, uint64_t world_size, pybind11::object& global_pg_nccl):
    num_experts(num_experts), num_max_dispatch_tokens_per_rank(num_max_dispatch_tokens_per_rank), khidden(khidden), hidden_size(hidden_size), num_tokens(num_tokens), 
    num_topk(num_topk), world_size(world_size) {
        auto* pg_nccl = reinterpret_cast<ProcessGroupNCCL*>(global_pg_nccl.ptr());
        this->global_pg = c10::intrusive_ptr<ProcessGroupNCCL>::reclaim(pg_nccl);
}

void Tool::create_mode(int num_splits) {
    this->multi_token_moe = std::make_shared<MultiTokenMoE>(num_experts, num_max_dispatch_tokens_per_rank, khidden, hidden_size, num_tokens, num_topk, world_size, global_pg, false/*enable_random*/, num_splits);
}

void Tool::load_weights(std::tuple<torch::Tensor, torch::Tensor> w13_weight, std::tuple<torch::Tensor, torch::Tensor> w2_weight) {
    this->multi_token_moe->load_weights(w13_weight, w2_weight);
}

void Tool::get_split_metadata(std::vector<uint64_t> num_split_tokens) {
    this->multi_token_moe->get_split_metadata(num_split_tokens);
}

void Tool::load_inputs_and_split(torch::Tensor hidden_states_in, torch::Tensor topk_ids_in, torch::Tensor topk_weights_in) {
    this->multi_token_moe->load_inputs_and_split(hidden_states_in, topk_ids_in, topk_weights_in);
}

void Tool::launch(std::string launch_mode, int deep_sms) {
    std::shared_ptr<FUSEConfig> fuse_config = std::make_shared<FUSEConfig>(getSmCount() - deep_sms, deep_sms);
    if (launch_mode == "sched") {
        this->multi_token_moe->launch(LaunchMode::SCHED_LAUNCH, fuse_config);
    } else if (launch_mode == "sync") {
        this->multi_token_moe->launch(LaunchMode::SYNC_LAUNCH, fuse_config);
    } else {
        throw std::runtime_error("Invalid launch mode");
    }
}

torch::Tensor Tool::get_merged_output() {
    return this->multi_token_moe->get_merged_output();
}
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "DeepFUSE: an efficient moe library";

    pybind11::class_<deep_fuse::Tool>(m, "Tool")
        .def(pybind11::init<uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, pybind11::object&>())
        .def("create_mode", &deep_fuse::Tool::create_mode)
        .def("load_weights", &deep_fuse::Tool::load_weights)
        .def("get_split_metadata", &deep_fuse::Tool::get_split_metadata)
        .def("load_inputs_and_split", &deep_fuse::Tool::load_inputs_and_split)
        .def("launch", &deep_fuse::Tool::launch)
        .def("get_merged_output", &deep_fuse::Tool::get_merged_output);
}