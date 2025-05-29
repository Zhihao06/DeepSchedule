#pragma once

void launch_gemm(void* __raw_lhs, void* __raw_lhs_scales, void* __raw_rhs, void* __raw_rhs_scales, void* __raw_out, void* __raw_grouped_layout, int m, void* __raw_stream, int num_sms);
void launch_gemm_m4_n4096_k1536_group16_sm78(void* __raw_lhs, void* __raw_lhs_scales, void* __raw_rhs, void* __raw_rhs_scales, void* __raw_out, void* __raw_grouped_layout, int m, void* __raw_stream, int num_sms);
void launch_gemm_m4_n3072_k4096_group16_sm78(void* __raw_lhs, void* __raw_lhs_scales, void* __raw_rhs, void* __raw_rhs_scales, void* __raw_out, void* __raw_grouped_layout, int m, void* __raw_stream, int num_sms);
void launch_gemm_m4_n2048_k1536_group16_sm78(void* __raw_lhs, void* __raw_lhs_scales, void* __raw_rhs, void* __raw_rhs_scales, void* __raw_out, void* __raw_grouped_layout, int m, void* __raw_stream, int num_sms);
void launch_gemm_m4_n3072_k2048_group16_sm78(void* __raw_lhs, void* __raw_lhs_scales, void* __raw_rhs, void* __raw_rhs_scales, void* __raw_out, void* __raw_grouped_layout, int m, void* __raw_stream, int num_sms);
