#pragma once

#include "configs.cuh"

#ifndef SETUP_LAUNCH_CONFIG
#define SETUP_LAUNCH_CONFIG(num_sms, num_threads, stream) \
    cudaLaunchConfig_t cfg = {(num_sms), (num_threads), 0, stream, nullptr, 0}; \
    cudaLaunchAttribute attr[1]; \
    attr[0].id = cudaLaunchAttributeCooperative; \
    attr[0].val.cooperative = 1; \
    cfg.attrs = attr; \
    cfg.numAttrs = 1
#endif

#ifndef LAUNCH_KERNEL
#define LAUNCH_KERNEL(config, kernel, ...) CUDA_CHECK(cudaLaunchKernelEx(config, kernel, ##__VA_ARGS__))
#endif

#define SWITCH_RANKS(case_macro) \
    switch (num_ranks) { \
        case 2: case_macro(2); \
        case 4: case_macro(4); \
        case 8: case_macro(8); \
        default: EP_HOST_ASSERT(false and "Unsupported ranks"); \
    } while (false)

#define SWITCH_RDMA_RANKS(case_macro) \
    switch (num_ranks / NUM_MAX_NVL_PEERS) { \
        case 2: case_macro(2); \
        case 3: case_macro(3); \
        case 4: case_macro(4); \
        case 8: case_macro(8); \
        case 16: case_macro(16); \
        case 18: case_macro(18); \
        case 20: case_macro(20); \
        default: EP_HOST_ASSERT(false and "Unsupported RDMA ranks"); \
    } while (false)

#define SWITCH_RANKS_WITH_DTYPE(dtype, case_macro) \
    switch (num_ranks) { \
        case 2: case_macro(dtype, 2); \
        case 4: case_macro(dtype, 4); \
        case 8: case_macro(dtype, 8); \
        default: EP_HOST_ASSERT(false && "Unsupported ranks"); \
    } while (false)

#define SWITCH_TYPES(case_macro) \
    switch (type) { \
        case CUDA_R_16BF: case_macro(nv_bfloat16); \
        case CUDA_R_32F:  case_macro(float); \
        default: EP_HOST_ASSERT(false && "Unsupported type"); \
    } while (false)

#define SWITCH_HIDDEN(inner_macro) \
    switch (hidden) { \
        case 2048: inner_macro(2048); break; \
        case 2560: inner_macro(2560); break; \
        case 3072: inner_macro(3072); break; \
        case 4096: inner_macro(4096); break; \
        case 5120: inner_macro(5120); break; \
        case 7168: inner_macro(7168); break; \
        case 8192: inner_macro(8192); break; \
        case 16384: inner_macro(16384); break; \
        default: EP_HOST_ASSERT(false && "Unsupported hidden"); \
    } while (false)

#define SWITCH_SMS(hidden_const, inner_macro) \
    do { \
        if (num_sms <= 4) { \
            inner_macro(hidden_const, 4); \
        } else if (num_sms <= 8) { \
            inner_macro(hidden_const, 8); \
        } else if (num_sms <= 12) { \
            inner_macro(hidden_const, 12); \
        } else if (num_sms <= 16) { \
            inner_macro(hidden_const, 16); \
        } else if (num_sms <= 20) { \
            inner_macro(hidden_const, 20); \
        } else if (num_sms <= 24) { \
            inner_macro(hidden_const, 24); \
        } else if (num_sms <= 28) { \
            inner_macro(hidden_const, 28); \
        } else if (num_sms <= 32) { \
            inner_macro(hidden_const, 32); \
        } else if (num_sms <= 36) { \
            inner_macro(hidden_const, 36); \
        } else if (num_sms <= 40) { \
            inner_macro(hidden_const, 40); \
        } else if (num_sms <= 44) { \
            inner_macro(hidden_const, 44); \
        } else if (num_sms <= 48) { \
            inner_macro(hidden_const, 48); \
        } else if (num_sms <= 52) { \
            inner_macro(hidden_const, 52); \
        } else if (num_sms <= 56) { \
            inner_macro(hidden_const, 56); \
        } else if (num_sms <= 60) { \
            inner_macro(hidden_const, 60); \
        } else if (num_sms <= 64) { \
            inner_macro(hidden_const, 64); \
        } else if (num_sms <= 68) { \
            inner_macro(hidden_const, 68); \
        } else if (num_sms <= 72) { \
            inner_macro(hidden_const, 72); \
        } else if (num_sms <= 76) { \
            inner_macro(hidden_const, 76); \
        } else if (num_sms <= 80) { \
            inner_macro(hidden_const, 80); \
        } else if (num_sms <= 84) { \
            inner_macro(hidden_const, 84); \
        } else if (num_sms <= 88) { \
            inner_macro(hidden_const, 88); \
        } else if (num_sms <= 92) { \
            inner_macro(hidden_const, 92); \
        } else if (num_sms <= 96) { \
            inner_macro(hidden_const, 96); \
        } else if (num_sms <= 100) { \
            inner_macro(hidden_const, 100); \
        } else if (num_sms <= 104) { \
            inner_macro(hidden_const, 104); \
        } else if (num_sms <= 108) { \
            inner_macro(hidden_const, 108); \
        } else if (num_sms <= 112) { \
            inner_macro(hidden_const, 112); \
        } else if (num_sms <= 116) { \
            inner_macro(hidden_const, 116); \
        } else if (num_sms <= 120) { \
            inner_macro(hidden_const, 120); \
        } else if (num_sms <= 124) { \
            inner_macro(hidden_const, 124); \
        } else if (num_sms <= 128) { \
            inner_macro(hidden_const, 128); \
        } else if (num_sms <= 132) { \
            inner_macro(hidden_const, 132); \
        } else { \
            EP_HOST_ASSERT(false && "num_sms too large"); \
        } \
    } while (false)

#define SWITCH_EXPERTS(hidden_const, num_sms_const, case_macro) \
    do { \
        if (num_experts <= num_sms_const) { \
            case_macro(hidden_const, num_sms_const, num_experts, 1, 32); \
        } else if (num_experts <= num_sms_const * 2) { \
            case_macro(hidden_const, num_sms_const, num_experts, 2, 16); \
        } else if (num_experts <= num_sms_const * 3) { \
            case_macro(hidden_const, num_sms_const, num_experts, 3, 10); \
        } else if (num_experts <= num_sms_const * 4) { \
            case_macro(hidden_const, num_sms_const, num_experts, 4, 8); \
        } else if (num_experts <= num_sms_const * 5) { \
            case_macro(hidden_const, num_sms_const, num_experts, 5, 6); \
        } else if (num_experts <= num_sms_const * 6) { \
            case_macro(hidden_const, num_sms_const, num_experts, 6, 5); \
        } else if (num_experts <= num_sms_const * 7) { \
            case_macro(hidden_const, num_sms_const, num_experts, 7, 4); \
        } else if (num_experts <= num_sms_const * 8) { \
            case_macro(hidden_const, num_sms_const, num_experts, 8, 4); \
        } else if (num_experts <= num_sms_const * 9) { \
            case_macro(hidden_const, num_sms_const, num_experts, 9, 3); \
        } else if (num_experts <= num_sms_const * 10) { \
            case_macro(hidden_const, num_sms_const, num_experts, 10, 3); \
        } else if (num_experts <= num_sms_const * 14) { \
            case_macro(hidden_const, num_sms_const, num_experts, 14, 2); \
        } else { \
            EP_HOST_ASSERT(false && "Unsupported expert"); \
        } \
    } while (false)
