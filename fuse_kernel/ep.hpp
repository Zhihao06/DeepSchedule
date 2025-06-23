#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDADataType.h>
#include <torch/torch.h>
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/TCPStore.hpp>
#include <pybind11/functional.h>
#include <cassert>

#include "deep_ep.hpp"
#include "config.hpp"
#include "src/tools/data.hpp"
#include "src/tools/utils.hpp"
#include "src/tools/debug.hpp"
#include "src/tools/config.hpp"
#include "src/gemm_gen.hpp"

#include "src/moe/base_moe.hpp"
#include "src/moe/sequence_moe.hpp"
#include "src/moe/overlap_moe.hpp"
#include "src/moe/tbo_moe.hpp"
#include "src/moe/multi_token_moe.hpp"

using namespace deep_ep;
using namespace c10d;

#define TEMPLATE_SMS 54

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

void ep_moe(uint64_t num_experts, uint64_t num_max_dispatch_tokens_per_rank, uint64_t khidden, uint64_t hidden_size, uint64_t num_tokens, uint64_t num_topk, uint64_t world_size, ModeType mode) {
    global_pg->barrier()->wait();
    uint64_t num_groups = num_experts / world_size;
    std::vector<int> ep_sms = {24};
    int repeat_times = 10;

    if (mode == ModeType::NORMAL) {
        SequenceMoE moe(num_experts, num_max_dispatch_tokens_per_rank, khidden, hidden_size, num_tokens, num_topk,
            global_pg->getSize(), global_pg);
        moe.run(ep_sms, repeat_times, false/*enable_profile*/);
    } else if (mode == ModeType::OVERLAP) {
        OverlapMoE moe(num_experts, num_max_dispatch_tokens_per_rank, khidden, hidden_size, num_tokens, num_topk,
            global_pg->getSize(), global_pg);
        moe.run(ep_sms, repeat_times, false/*enable_profile*/);
    } else if (mode == ModeType::TBO) {
        TBOMoE moe(num_experts, num_max_dispatch_tokens_per_rank, khidden, hidden_size, num_tokens, num_topk,
            global_pg->getSize(), global_pg);
        moe.run(ep_sms, repeat_times, false/*enable_profile*/);
    } else if (mode == ModeType::MULTI_TOKEN) {
        MultiTokenMoE moe(num_experts, num_max_dispatch_tokens_per_rank, khidden, hidden_size, num_tokens, num_topk,
            global_pg->getSize(), global_pg, 8/*num_splits*/, {}/*num_split_tokens*/);
        moe.run(ep_sms, repeat_times, false/*enable_profile*/);
    } else {
        throw std::runtime_error("Not supported mode");
    }

    // // Prepare Cuda Graph
    // cudaGraph_t graph;
    // cudaGraphExec_t instance;
    // cudaStreamBeginCapture(current_stream, cudaStreamCaptureModeGlobal);
    // ep_moe_core_(num_experts, num_max_dispatch_tokens_per_rank, khidden, hidden_size, num_tokens,
    //     num_topk, world_size, num_groups, expected_m, m_max, current_stream, 
    //     buffer, 
    //     hidden_states,
    //     topk_ids,
    //     topk_weights,
    //     x_fp8,
    //     y_fp8,
    //     out,
    //     x_fp8_2,
    //     y_fp8_2,
    //     out_2);
    // cudaStreamEndCapture(current_stream, &graph);

    // // Instantiate Graph
    // cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);

    // // Execute Graph
    // for (auto i=0; i<10; i++) {
    //     cudaGraphLaunch(instance, current_stream);
    //     DEBUG_FILE();
    // }

    // // Release Resource
    // cudaGraphExecDestroy(instance);
    // cudaGraphDestroy(graph);
}