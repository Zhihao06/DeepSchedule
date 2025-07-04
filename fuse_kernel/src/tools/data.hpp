#pragma once

#include <torch/types.h>
#include <torch/python.h>
#include <torch/torch.h>
#include <vector>
#include <tuple>
#include "utils.hpp"

// Helper function for ceiling division
int64_t ceil_div(int64_t a, int64_t b) {
    return (a + b - 1) / b;
}

int64_t get_tma_aligned_size(int64_t size, int64_t elem_size) {
    // TMA 要求内存按 16 字节对齐
    const int64_t kAlignment = 16;
    TORCH_CHECK(kAlignment % elem_size == 0);
    int64_t aligned_size = ((size * elem_size + kAlignment - 1) / kAlignment) * kAlignment / elem_size;
    return aligned_size;
}

torch::Tensor get_col_major_tma_aligned_tensor(torch::Tensor x) {
    TORCH_CHECK(x.dim() == 2 || x.dim() == 3, "Input tensor must be 2D or 3D");

    bool remove_dim = false;
    int64_t m = x.size(-2);
    int64_t n = x.size(-1);
    int64_t elem_size = x.element_size();
    int64_t aligned_m = get_tma_aligned_size(m, elem_size);

    if (x.dim() == 2) {
        // Handle 2D case: (M, N)
        if (x.stride(0) == 1 && x.stride(1) == aligned_m) {
            return x;
        }
        x = x.unsqueeze(0);  // Add batch dim -> (1, M, N)
        remove_dim = true;
    }

    int64_t b = x.size(0);

    // Check if already in column-major and TMA-aligned format
    std::vector<int64_t> strides(x.strides().begin(), x.strides().end());
    if (strides[0] == aligned_m * n && strides[1] == 1 && strides[2] == aligned_m) {
        if (remove_dim) {
            return x.squeeze(0);
        } else {
            return x;
        }
    }

    // Create aligned tensor with transposed layout: (B, N, aligned_m)
    torch::Tensor aligned_x = torch::empty({b, n, aligned_m}, x.options());
    aligned_x = aligned_x.transpose(1, 2);  // Now shaped as (B, aligned_m, n)

    // Copy original data into aligned_x
    aligned_x.index_put_(
        {torch::indexing::Slice(), torch::indexing::Slice(0, m), torch::indexing::Slice()},
        x
    );

    aligned_x = aligned_x.index({torch::indexing::Slice(), torch::indexing::Slice(0, m), torch::indexing::Slice()});

    // Squeeze batch dimension if needed
    if (remove_dim) {
        return aligned_x.squeeze(0);
    } else {
        return aligned_x;
    }
}

std::tuple<torch::Tensor, torch::Tensor> per_block_cast_to_fp8(torch::Tensor x) {
    TORCH_CHECK(x.dim() == 2, "Input tensor must be 2D");
    int64_t m = x.size(0);
    int64_t n = x.size(1);

    int64_t padded_m = ceil_div(m, 128) * 128;
    int64_t padded_n = ceil_div(n, 128) * 128;

    torch::Tensor x_padded = torch::zeros({padded_m, padded_n}, x.options());
    x_padded.index_put_({torch::indexing::Slice(0, m), torch::indexing::Slice(0, n)}, x);

    int64_t block_size = 128;
    std::vector<int64_t> view_shape = {-1, block_size, padded_n / block_size, block_size};
    torch::Tensor x_view = x_padded.view(view_shape);

    torch::Tensor x_abs = x_view.abs().to(torch::kFloat32);
    torch::Tensor x_amax = x_abs.amax({1, 3}, /*keepdim=*/true).clamp_min_(1e-4);

    torch::Tensor x_scaled = (x_view.to(torch::kFloat32) * (448.0f / x_amax)).to(torch::kFloat8_e4m3fn);
    torch::Tensor result = x_scaled.view({padded_m, padded_n}).index({torch::indexing::Slice(0, m), torch::indexing::Slice(0, n)}).contiguous();

    torch::Tensor scale_factors = (x_amax / 448.0f).view({x_view.size(0), x_view.size(2)});

    return std::make_tuple(result, scale_factors);
}

std::tuple<torch::Tensor, torch::Tensor> per_token_cast_to_fp8(torch::Tensor x) {
    TORCH_CHECK(x.dim() == 2 && x.size(1) % 128 == 0,
                "Input must be 2D and second dimension divisible by 128");

    int64_t m = x.size(0);
    int64_t n = x.size(1);

    std::vector<int64_t> view_shape = {m, -1, 128};
    torch::Tensor x_view = x.view(view_shape);

    torch::Tensor x_abs = x_view.abs().to(torch::kFloat32);
    torch::Tensor x_amax = x_abs.amax(/*dim=*/2).view({m, -1}).clamp_min_(1e-4);

    torch::Tensor scaled_x = (x_view.to(torch::kFloat32) * (448.0f / x_amax.unsqueeze(2))).to(torch::kFloat8_e4m3fn);
    torch::Tensor result = scaled_x.view({m, n});
    torch::Tensor scale_factors = (x_amax / 448.0f).view({m, -1});

    return std::make_tuple(result, scale_factors);
}


std::tuple<std::tuple<torch::Tensor, torch::Tensor>, 
            std::tuple<torch::Tensor, torch::Tensor>, torch::Tensor>
construct_masked_grouped(int64_t num_groups, int64_t m, int64_t k, int64_t n) {
    auto x = torch::randn({num_groups, m, k}, dtype(torch::kBFloat16).device(torch::kCUDA));
    auto y = torch::randn({num_groups, n, k}, dtype(torch::kBFloat16).device(torch::kCUDA));
    auto out = torch::empty({num_groups, m, n}, dtype(torch::kBFloat16).device(torch::kCUDA));
    TORCH_CHECK(m % 4 == 0, "TMA alignment error: m");
    auto x_fp8_data = torch::empty_like(x, dtype(torch::kFloat8_e4m3fn));
    auto y_fp8_data = torch::empty_like(y, dtype(torch::kFloat8_e4m3fn));
    auto x_scale = torch::empty({num_groups, m, k / 128}, dtype(torch::kFloat32).device(torch::kCUDA));
    auto y_scale = torch::empty({num_groups, (n + 127) / 128, k / 128}, dtype(torch::kFloat32).device(torch::kCUDA));

    for (int64_t i = 0; i < num_groups; ++i) {
        auto x_group = x.index({i});
        auto y_group = y.index({i});

        auto [x_casted, x_scale_i] = per_token_cast_to_fp8(x_group);
        auto [y_casted, y_scale_i] = per_block_cast_to_fp8(y_group);

        x_fp8_data.index_put_({i}, x_casted);
        x_scale.index_put_({i}, x_scale_i);

        y_fp8_data.index_put_({i}, y_casted);
        y_scale.index_put_({i}, y_scale_i);
    }

    x_scale = get_col_major_tma_aligned_tensor(x_scale);

    auto x_fp8 = std::make_tuple(x_fp8_data, x_scale);
    auto y_fp8 = std::make_tuple(y_fp8_data, y_scale);

    return std::make_tuple(x_fp8, y_fp8, out);
}

/*
    hidden_states,
    topk_ids,
    topk_weights,
    x_fp8,
    y_fp8,
    out,
    x_fp8_2,
    y_fp8_2,
    out_2,
*/
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
           std::tuple<torch::Tensor, torch::Tensor>, std::tuple<torch::Tensor, torch::Tensor>, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           std::tuple<torch::Tensor, torch::Tensor>, std::tuple<torch::Tensor, torch::Tensor>, torch::Tensor>
initialize_random_inputs(int64_t num_tokens, int64_t num_topk, int64_t num_groups, int64_t num_experts,
                      int64_t m_max, int64_t hidden_size, int64_t khidden) {

    // 1. hidden_states: [num_tokens, hidden_size]
    torch::Tensor hidden_states = torch::randn({num_tokens, hidden_size}, dtype(torch::kBFloat16).device(torch::kCUDA));

    // 2. topk_ids: [num_tokens, num_topk]
    torch::Tensor topk_ids = torch::arange(num_tokens * num_topk, dtype(torch::kInt64).device(torch::kCUDA));
    topk_ids = (topk_ids % num_experts).view({num_tokens, num_topk});

    // 3. topk_weights: [num_tokens, num_topk]
    torch::Tensor topk_weights = torch::randn({num_tokens, num_topk}, dtype(torch::kFloat32).device(torch::kCUDA)).abs();

    // 4. First call to construct_masked_grouped
    auto result1 = construct_masked_grouped(num_groups, m_max, hidden_size, khidden);
    auto x_fp8 = std::get<0>(result1);
    auto y_fp8 = std::get<1>(result1);
    auto out = std::get<2>(result1);

    // 5. silu_and_mul_masked inputs
    auto o_vec = torch::empty({num_groups, m_max, khidden / 2}, torch::kCUDA).to(torch::kFloat8_e4m3fn);
    auto o_scales = torch::empty({num_groups, std::max(static_cast<int64_t>(1), khidden / 256), m_max}, dtype(torch::kFloat32).device(torch::kCUDA));
    std::vector<int64_t> stride_0 = {num_groups, m_max, std::max(static_cast<int64_t>(1), khidden / 256)};
    std::vector<int64_t> stride_1 = {m_max * std::max(static_cast<int64_t>(1), khidden / 256), static_cast<int64_t>(1), m_max};
    auto o_scales_strided = torch::as_strided(o_scales, stride_0, stride_1);
    auto silu_out = torch::empty({0}, dtype(torch::kBFloat16).device(torch::kCUDA));

    // 6. Second call to construct_masked_grouped
    auto result2 = construct_masked_grouped(num_groups, m_max, khidden / 2, hidden_size);
    auto x_fp8_2 = std::get<0>(result2);
    auto y_fp8_2 = std::get<1>(result2);
    auto out_2 = std::get<2>(result2);

    // 7. Return all tensors as a tuple
    return std::make_tuple(
        hidden_states, topk_ids, topk_weights,
        x_fp8, y_fp8, out,
        o_vec, o_scales, o_scales_strided, silu_out,
        x_fp8_2, y_fp8_2, out_2
    );
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
initialize_empty_intermediate(int64_t num_tokens, int64_t num_topk, int64_t num_groups, int64_t num_experts,
    int64_t m_max, int64_t hidden_size, int64_t khidden) {

    auto out = torch::empty({num_groups, m_max, khidden}, dtype(torch::kBFloat16).device(torch::kCUDA));
    
    // 5. silu_and_mul_masked inputs
    auto o_vec = torch::empty({num_groups, m_max, khidden / 2}, torch::kCUDA).to(torch::kFloat8_e4m3fn);
    auto o_scales = torch::empty({num_groups, std::max(static_cast<int64_t>(1), khidden / 256), m_max}, dtype(torch::kFloat32).device(torch::kCUDA));
    std::vector<int64_t> stride_0 = {num_groups, m_max, std::max(static_cast<int64_t>(1), khidden / 256)};
    std::vector<int64_t> stride_1 = {m_max * std::max(static_cast<int64_t>(1), khidden / 256), static_cast<int64_t>(1), m_max};
    auto o_scales_strided = torch::as_strided(o_scales, stride_0, stride_1);
    auto silu_out = torch::empty({0}, dtype(torch::kBFloat16).device(torch::kCUDA));

    auto out_2 = torch::empty({num_groups, m_max, hidden_size}, dtype(torch::kBFloat16).device(torch::kCUDA));

    return std::make_tuple(out, o_vec, o_scales, o_scales_strided, silu_out, out_2);
}

std::vector<torch::Tensor> custom_split(const torch::Tensor& tensor, const std::vector<uint64_t>& sizes, int64_t dim) {
    std::vector<torch::Tensor> result;
    int offset = 0;
    for (int size : sizes) {
        torch::Tensor slice = tensor.slice(dim, offset, offset + size);
        result.push_back(slice);
        offset += size;
    }
    return result;
}