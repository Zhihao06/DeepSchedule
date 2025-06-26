#pragma once

#define LAUNCH_GEMM(n, k) launch_gemm_m4_n##n##_k##k##_group16_sm78

struct FUSEConfig {
    int gemm_sms;
    int ep_sms;

    FUSEConfig(int gemm_sms, int ep_sms): 
        gemm_sms(gemm_sms), ep_sms(ep_sms) {};
};

// run mode
enum ModeType {
    NORMAL,
    OVERLAP,
    TBO,
    MULTI_TOKEN
};

// launch mode
enum LaunchMode {
    SYNC_LAUNCH,
    SCHED_LAUNCH
};