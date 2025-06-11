#pragma once

__device__ __forceinline__ void kernel_timer(unsigned long long *out, int sm_id, int thread_id) {
    if (sm_id == 0 and thread_id == 0) {
        asm volatile ("mov.u64 %0, %%clock64;" : "=l"(*out));
    }
    return;
}