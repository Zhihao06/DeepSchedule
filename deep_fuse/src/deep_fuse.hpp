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
    Tool(uint64_t num_experts, uint64_t num_max_dispatch_tokens_per_rank, uint64_t khidden, uint64_t hidden_size, uint64_t num_tokens, uint64_t num_topk, uint64_t world_size, const pybind11::object& global_pg_nccl);
    void create_mode(int num_splits);
    void load_weights(const torch::Tensor& w13_weight_data, const torch::Tensor& w13_weight_scale, const torch::Tensor& w2_weight_data, const torch::Tensor& w2_weight_scale);
    void get_metadata(const std::string& mode, uint64_t num_tokens, const std::vector<uint64_t>& num_split_tokens);
    void load_inputs(const std::string& mode, const torch::Tensor& hidden_states_in, const torch::Tensor& topk_ids_in, const torch::Tensor& topk_weights_in);
    void launch(const std::string& mode, const std::string& launch_mode, int deepep_sms);
    torch::Tensor get_merged_output(const std::string& mode);
    std::tuple<torch::Tensor, std::optional<torch::Tensor>, torch::Tensor, uint64_t> low_latency_dispatch_interface(const std::string& mode, int deepep_sms);
    torch::Tensor low_latency_combine_interface(const std::string& mode, const torch::Tensor& compute_result, int deepep_sms);
};
}