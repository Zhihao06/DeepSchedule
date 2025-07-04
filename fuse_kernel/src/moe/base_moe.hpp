#pragma once

#include "../ops/common.h"
#include "../tools/buffer.hpp"

using namespace deep_ep;
using namespace c10d;

class BaseMoE {
protected:
    uint64_t num_tokens, num_topk, num_groups, num_experts, m_max, hidden_size, khidden, num_max_dispatch_tokens_per_rank, expected_m, world_size;

    virtual void _moe_core(std::shared_ptr<FUSEConfig>& fuse_config, LaunchMode launch_mode, bool enable_profile) = 0;

public:
    torch::Tensor hidden_states, topk_ids, topk_weights;
    std::tuple<torch::Tensor, torch::Tensor> x_fp8, y_fp8;
    torch::Tensor o_vec, o_scales, o_scales_strided, silu_out;
    std::tuple<torch::Tensor, torch::Tensor> x_fp8_2, y_fp8_2;
    torch::Tensor out, out_2;

    c10::intrusive_ptr<ProcessGroupNCCL> global_pg;
    std::shared_ptr<Buffer> buffer;

    BaseMoE(uint64_t num_experts, uint64_t num_max_dispatch_tokens_per_rank, uint64_t khidden, uint64_t hidden_size, uint64_t num_tokens, 
        uint64_t num_topk, uint64_t world_size, c10::intrusive_ptr<ProcessGroupNCCL>& global_pg):
        num_experts(num_experts), num_max_dispatch_tokens_per_rank(num_max_dispatch_tokens_per_rank), 
        khidden(khidden), hidden_size(hidden_size), num_tokens(num_tokens), num_topk(num_topk), 
        world_size(world_size), global_pg(global_pg) {
        num_groups = num_experts / world_size;
        expected_m = std::max(1UL, (num_tokens * num_topk + num_experts - 1) / num_experts * 2);
        m_max = num_max_dispatch_tokens_per_rank * world_size;
    }

    ~BaseMoE() {
    }

    void run(std::vector<int> ep_sms, LaunchMode launch_mode, int repeat_times, bool enable_profile) {
        for (auto ep_sm: ep_sms) {
            std::shared_ptr<FUSEConfig> fuse_config = std::make_shared<FUSEConfig>(78-ep_sm, ep_sm);
            for (int i = 0; i < repeat_times; i++) {
                _moe_core(fuse_config, launch_mode, enable_profile);
            }
        }
    }
};