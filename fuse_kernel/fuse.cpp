#include <cuda.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <iostream>
#include "ep.hpp"
#include "src/tools/config.hpp"

void process_args(int argc, char** argv, ModeType& mode, bool& enable_traverse) {
    if (argc <= 1) {
        std::cerr << "No parameters provided. Using default values." << std::endl;
        return;
    }
    if (argc >= 2) {
        std::string arg = argv[1];
        if (arg == "0" || arg == "normal") {
            mode = ModeType::NORMAL;
        } else if (arg == "1" || arg == "overlap") {
            mode = ModeType::OVERLAP;
        } else if (arg == "2" || arg == "tbo") {
            mode = ModeType::TBO;
        } else {
            std::cerr << "Usage: " << argv[0] << " [0|1|2|normal|overlap|tbo]" << std::endl;
        }
    } 
    if (argc >= 3) {
        std::string arg = argv[2];
        if (arg == "0" || arg == "false" || arg == "False") {
            enable_traverse = false;
        } else if (arg == "1" || arg == "true" || arg == "True") {
            enable_traverse = true;
        } else {
            std::cerr << "Usage: " << argv[0] << " [0|1|true|True|false|False]" << std::endl;
        }
    }
}

void ep_moe_traverse(int rank, int world_size, ModeType mode) {
    std::vector<uint64_t> num_tokens_l = {32, 64, 128, 256, 512};
    std::vector<uint64_t> num_experts_l = {32, 64, 128};
    std::vector<uint64_t> hidden_l = {4096};
    std::vector<uint64_t> khidden_l = {3072};
    for (auto num_tokens: num_tokens_l) {
        for (auto num_experts: num_experts_l) {
            for (auto hidden: hidden_l) {
                for (auto khidden: khidden_l) {
                    if (rank == 0) {
                        std::cout << "num_experts: " << num_experts << ", num_tokens: " << num_tokens << ", hidden: " << hidden << ", khidden: " << khidden << std::endl;
                    }
                    ep_moe(num_experts, 1024/*num_max_dispatch_tokens_per_rank*/, khidden, hidden, num_tokens, 8/*num_topk*/, world_size, mode);
                }
            }
        }
    }
}

int main(int argc, char** argv) {
    const char* env_rank = std::getenv("OMPI_COMM_WORLD_RANK");
    const char* env_world_size = std::getenv("OMPI_COMM_WORLD_SIZE");

    int rank = env_rank ? std::atoi(env_rank) : -1;
    int world_size = env_world_size ? std::atoi(env_world_size) : -1;

    ModeType mode = ModeType::NORMAL;
    bool enable_traverse = false;
    process_args(argc, argv, mode, enable_traverse);

    init_distributed(rank, world_size);
    CUDA_CHECK(cudaSetDevice(rank));
    
    if (enable_traverse) {
        ep_moe_traverse(rank, world_size, mode);
    } else {
        ep_moe(
            128/*num_experts*/, 
            1024/*num_max_dispatch_tokens_per_rank*/, 
            3072/*khidden*/, 
            4096/*hidden_size*/, 
            64/*num_tokens*/, 
            8/*num_topk*/, 
            world_size,
            mode);
    }

    destroy_distributed();
    return 0;
}