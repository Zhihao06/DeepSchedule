#pragma once

template <uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t BLOCK_N_PADDING,
          uint32_t kSwizzleDMode,
          uint32_t kNumGroups, uint32_t kNumStages,
          uint32_t kNumTMAMulticast, bool kIsTMAMulticastOnA,
          uint32_t SMEM_SIZE>
void launch_gemm(void* __raw_lhs, void* __raw_lhs_scales, void* __raw_rhs, void* __raw_rhs_scales, void* __raw_out, void* __raw_grouped_layout, int m, void* __raw_stream, int num_sms);

