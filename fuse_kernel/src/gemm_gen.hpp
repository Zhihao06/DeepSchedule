
#pragma once

#include <stdexcept>
#include "gemm.cuh"

using GemmLauncherFunc = void (*)(void*, void*, void*, void*, void*, void*, int, void*, int);

GemmLauncherFunc get_function_for_gemm(int num_tokens, int hidden, int intermediate, int num_groups, int sms) {
    /*
        use gemm_codegen.py to get code like:
        if (num_tokens == 32 and hidden == 3072 and intermediate == 4096 and num_groups == 16 and sms == 54) return launch_gemm<3072, 4096, 64, 160, 128, 0, 64, 16, 4, 1, true, 136512>;
        else if (num_tokens == 32 and hidden == 4096 and intermediate == 1536 and num_groups == 16 and sms == 54) return launch_gemm<4096, 1536, 64, 160, 128, 0, 64, 16, 4, 1, true, 136352>;
        else throw std::runtime_error("unsupported parameters for gemm!");
    */
    throw std::runtime_error("unsupported parameters for gemm!");
}