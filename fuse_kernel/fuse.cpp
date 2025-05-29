#include <cuda.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <iostream>
#include "ep.hpp"

int main() {
    const char* env_rank = std::getenv("OMPI_COMM_WORLD_RANK");
    const char* env_world_size = std::getenv("OMPI_COMM_WORLD_SIZE");

    int rank = env_rank ? std::atoi(env_rank) : -1;
    int world_size = env_world_size ? std::atoi(env_world_size) : -1;

    init_distributed(rank, world_size);
    CUDA_CHECK(cudaSetDevice(rank));
    
    ep_moe(
        128/*num_experts*/, 
        1024/*num_max_dispatch_tokens_per_rank*/, 
        3072/*khidden*/, 
        4096/*hidden_size*/, 
        32/*num_tokens*/, 
        8/*num_topk*/, 
        world_size);

    destroy_distributed();
    return 0;
}