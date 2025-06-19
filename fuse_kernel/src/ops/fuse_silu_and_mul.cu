#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include "common.h"
#include <cuda_bf16.h>
#include <cuda_fp8.h>

template <typename TO, typename FROM>
__device__ __forceinline__ TO to(FROM val) {
  return to(val);
}
template <>
__device__ __forceinline__ float to(float val) {
  return val;
}
template <>
__device__ __forceinline__ float to(__half val) {
  return __half2float(val);
}
template <>
__device__ __forceinline__ __half to(float val) {
  return __float2half(val);
}
template <typename T>
inline __device__ T ldg(const T* val) {
  return LDG(val);
}

#ifdef ENABLE_BF16
template <>
__device__ __forceinline__ float to(__nv_bfloat16 val) {
  return __bfloat162float(val);
}
template <>
__device__ __forceinline__ __nv_bfloat16 to(float val) {
  return __float2bfloat16(val);
}
template <>
inline __device__ __nv_bfloat16 ldg(const __nv_bfloat16* val) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  return val[0];
#else
  return LDG(val);
#endif
}
#endif
  
__device__ __forceinline__ float silu(const float& x) {
  return (x / (1.0f + expf(-x)));
}

__forceinline__ __device__ float half_warp_reduce_max(float value) {
    auto mask = __activemask();
    // The mask be in `{0xffffffff, 0xffff}`
    value = max(value, __shfl_xor_sync(mask, value, 8));
    value = max(value, __shfl_xor_sync(mask, value, 4));
    value = max(value, __shfl_xor_sync(mask, value, 2));
    value = max(value, __shfl_xor_sync(mask, value, 1));
    return value;
}

__forceinline__ __device__ int get_lane_id() {
    int lane_id;
    asm("mov.s32 %0, %laneid;" : "=r"(lane_id));
    return lane_id;
}

template <typename T>
__global__ void fuse_silu_and_mul_masked_kernel(
    T* __restrict__ out, // [..., d]
    void* o_vec,
    void* o_scales,
    const T* __restrict__ input, // [..., 2, d]
    const int32_t* counts,
    const int max_tokens_per_block,
    const int num_input_tokens,
    const int dim) {
  const int lane_id = get_lane_id();

  const int kNumElemsPerRead = 8;
  const int n_elems = dim / kNumElemsPerRead;
  constexpr float kFP8Margin = 1e-4, kFP8Amax = 448, kFP8AmaxInv = 1.0f / 448.0f;

  int expert_id = blockIdx.x;
  for (int64_t token_idx = blockIdx.y; token_idx < max_tokens_per_block; token_idx += gridDim.y) { 

    if (token_idx >= *(counts + expert_id)) 
      break;

    const int64_t global_offset = expert_id * max_tokens_per_block + token_idx;

    // NOTE(zycao): Do Silu and FP8 quantization. Quant codes copied from DeepEP internode_ll.cu.
    const auto x_int4 = reinterpret_cast<const int4*>(input) + global_offset * n_elems * 2;
    const auto y_int4 = reinterpret_cast<const int4*>(input) + global_offset * n_elems * 2 + n_elems;
    int4* o_int4 = out ? reinterpret_cast<int4*>(out) + global_offset * n_elems : nullptr;
    auto x_vec = reinterpret_cast<int64_t*>(o_vec) + global_offset * n_elems;
    auto x_scales = reinterpret_cast<float*>(o_scales) + expert_id * max_tokens_per_block * dim / 128;

    for (int64_t i = threadIdx.x; i < n_elems; i += blockDim.x) {
      // Read
      auto x_int4_value = __ldg(x_int4 + i);
      auto y_int4_value = __ldg(y_int4 + i);
      auto x_values = reinterpret_cast<T*>(&x_int4_value);
      auto y_values = reinterpret_cast<T*>(&y_int4_value);

      int4 z_int4_value;
      auto z_values = reinterpret_cast<T*>(&z_int4_value);

      float fp32_values[kNumElemsPerRead];
      float amax = kFP8Margin, scale, scale_inv;

      #pragma unroll
      for (int j = 0; j < kNumElemsPerRead; ++j) {
        // Silu and mul.
        z_values[j] = to<T>(silu(to<float>(x_values[j])) * to<float>(y_values[j]));

        // Calculate local amax
        fp32_values[j] = static_cast<float>(z_values[j]);
        amax = fmaxf(amax, fabsf(fp32_values[j]));
      }

      if (o_int4 != nullptr) o_int4[i] = z_int4_value;
      
      // Reduce amax and scale
      amax = half_warp_reduce_max(amax), scale = kFP8Amax / amax, scale_inv = amax * kFP8AmaxInv;
      if (lane_id == 0 or lane_id == 16)
          x_scales[i * kNumElemsPerRead / 128 * max_tokens_per_block + token_idx] = scale_inv;

      // Cast into x_vec
      int64_t int2_value;
      auto fp8x2_values = reinterpret_cast<__nv_fp8x2_storage_t*>(&int2_value);
      #pragma unroll
      for (int j = 0; j < kNumElemsPerRead; j += 2) {
          float2 fp32x2 = {fp32_values[j] * scale, fp32_values[j + 1] * scale};
          fp8x2_values[j / 2] = __nv_cvt_float2_to_fp8x2(fp32x2, __NV_SATFINITE, __NV_E4M3);
      }
      x_vec[i] = int2_value;
    }
  }
}

template <typename scalar_t>
void fuse_silu_and_mul_masked_launcher(
    torch::Tensor& out, torch::Tensor& o_vec, torch::Tensor& o_scales, torch::Tensor& input, torch::Tensor& counts, const int max_tokens_per_block) {
  int64_t num_input_tokens = input.numel() / input.size(-1);
  scalar_t* output_ptr = out.numel() == 0 ? nullptr : reinterpret_cast<scalar_t*>(out.data_ptr());
  assert (num_input_tokens = counts.numel() * max_tokens_per_block);

  int dim = input.size(-1) / 2;

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));

  int64_t sub_channels = 256; // divide max_tokens_per_block into 256 sub-channels.
  int block_size = std::min(dim / 8, 1024);
  
  dim3 grid(counts.numel(), sub_channels);
  dim3 block(block_size);

  assert (dim % 128 == 0);

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  fuse_silu_and_mul_masked_kernel<scalar_t><<<grid, block, 0, stream>>>(
    output_ptr,
    reinterpret_cast<void*>(o_vec.data_ptr()),
    reinterpret_cast<void*>(o_scales.data_ptr()),
    reinterpret_cast<scalar_t*>(input.data_ptr()),
    reinterpret_cast<int32_t*>(counts.data_ptr()),
    max_tokens_per_block,
    num_input_tokens,
    dim);
}

void fuse_silu_and_mul_masked(
    torch::Tensor& out, torch::Tensor& o_vec, torch::Tensor& o_scales, torch::Tensor& input, torch::Tensor& counts, const int max_tokens_per_block) {
  std::uintptr_t input_addr = reinterpret_cast<std::uintptr_t>(input.data_ptr());
  std::uintptr_t out_addr = reinterpret_cast<std::uintptr_t>(out.data_ptr());
  int32_t dim = input.size(-1) / 2;
  int32_t ele_bytes = input.element_size();
  assert (ele_bytes == 2);

  if (input.scalar_type() == at::ScalarType::Half) {
      fuse_silu_and_mul_masked_launcher<half>(out, o_vec, o_scales, input, counts, max_tokens_per_block);
#ifdef ENABLE_BF16
  } else if (input.scalar_type() == at::ScalarType::BFloat16) {
      fuse_silu_and_mul_masked_launcher<__nv_bfloat16>(out, o_vec, o_scales, input, counts, max_tokens_per_block);
#endif
  } else {
    throw std::runtime_error(
        "Unsupported input dtype encountered, supported dtypes now are "
        "{Half, BFloat16}");
  }
}