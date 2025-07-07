#pragma once

namespace deep_fuse { 
struct __attribute__((visibility("default"))) Tool {

private:
    uint64_t num_experts;
    uint64_t num_max_dispatch_tokens_per_rank;
    uint64_t khidden;
    uint64_t hidden_size;
    uint64_t num_tokens;
    uint64_t num_topk;
    uint64_t world_size;
    c10::intrusive_ptr<ProcessGroupNCCL> global_pg;

    // moe layer ptr
    std::shared_ptr<SequenceMoE> sequence_moe;
    std::shared_ptr<OverlapMoE> overlap_moe;
    std::shared_ptr<TBOMoE> tbo_moe;
    std::shared_ptr<MultiTokenMoE> multi_token_moe;

public:
    Tool(uint64_t num_experts, uint64_t num_max_dispatch_tokens_per_rank, uint64_t khidden, uint64_t hidden_size, uint64_t num_tokens, uint64_t num_topk, uint64_t world_size, pybind11::object& global_pg_nccl);
    void create_mode(int num_splits);
    void load_weights(std::tuple<torch::Tensor, torch::Tensor> w13_weight, std::tuple<torch::Tensor, torch::Tensor> w2_weight);
    void get_split_metadata(uint64_t num_tokens, std::vector<uint64_t> num_split_tokens);
    void load_inputs_and_split(torch::Tensor hidden_states_in, torch::Tensor topk_ids_in, torch::Tensor topk_weights_in);
    void launch(std::string launch_mode, int deep_sms);
    torch::Tensor get_merged_output();
};
}