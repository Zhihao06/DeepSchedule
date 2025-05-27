
struct FUSEConfig {
    int gemm_sms;
    int ep_sms;

    FUSEConfig(int gemm_sms, int ep_sms): 
        gemm_sms(gemm_sms), ep_sms(ep_sms) {};
};