#pragma once

void fuse_silu_and_mul_masked(
    torch::Tensor& out,       // [..., d]
    torch::Tensor& o_vec,       // [..., d]
    torch::Tensor& o_scales,       // [..., d]
    torch::Tensor& input,    // [..., 2*d]
    torch::Tensor& counts,
    int max_tokens_per_block,
    std::optional<cudaStream_t> stream = std::nullopt);
