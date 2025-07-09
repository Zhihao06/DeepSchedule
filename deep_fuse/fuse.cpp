#include <cuda.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <iostream>
#include "ep.hpp"
#include "src/tools/config.hpp"

std::vector<uint64_t> splitStringToIntVector(const std::string& str, char delimiter = ',') {
    std::vector<uint64_t> result;
    std::stringstream ss(str);
    std::string token;

    while (std::getline(ss, token, delimiter)) {
        try {
            result.push_back(std::stoi(token));
        } catch (const std::invalid_argument& e) {
            std::cerr << "Invalid integer: " << token << std::endl;
        }
    }
    return result;
}

void process_args(int argc, char** argv, ModeType& mode, bool& enable_traverse, LaunchMode& launch_mode, std::vector<uint64_t>& num_splits) {
    if (argc <= 1) {
        std::cerr << "No parameters provided. Using default values." << std::endl;
        return;
    }
    if (argc >= 2) { // mode
        std::string arg = argv[1];
        if (arg == "0" || arg == "normal") {
            mode = ModeType::NORMAL;
        } else if (arg == "1" || arg == "overlap") {
            mode = ModeType::OVERLAP;
        } else if (arg == "2" || arg == "tbo") {
            mode = ModeType::TBO;
        } else if (arg == "3" || arg == "multitoken") {
            mode = ModeType::MULTI_TOKEN;
        } else {
            std::cerr << "Usage: " << argv[0] << " [0|1|2|3|normal|overlap|tbo|multitoken]" << std::endl;
        }
    } 
    if (argc >= 3) { // enable_traverse
        std::string arg = argv[2];
        if (arg == "0" || arg == "false" || arg == "False") {
            enable_traverse = false;
        } else if (arg == "1" || arg == "true" || arg == "True") {
            enable_traverse = true;
        } else {
            std::cerr << "Usage: " << argv[0] << " [0|1|true|True|false|False]" << std::endl;
        }
    }
    if (argc >= 4) { // launch_mode sync, sched
        std::string arg = argv[3];
        if (arg == "0" || arg == "sync") {
            launch_mode = LaunchMode::SYNC_LAUNCH;
        } else if (arg == "1" || arg == "sched") {
            launch_mode = LaunchMode::SCHED_LAUNCH;
        } else {
            std::cerr << "Usage: " << argv[0] << " [0|1|sync|sched]" << std::endl;
        }
    }
    if (argc >= 5) {
        std::string arg = argv[4];
        num_splits = splitStringToIntVector(arg);
    }
}

void ep_moe_traverse(int rank, int world_size, ModeType mode, LaunchMode launch_mode, std::vector<uint64_t> num_splits) {
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
                    ep_moe(num_experts, 1024/*num_max_dispatch_tokens_per_rank*/, khidden, hidden, num_tokens, 8/*num_topk*/, world_size, mode, launch_mode, num_splits);
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
    LaunchMode launch_mode = LaunchMode::SYNC_LAUNCH;
    std::vector<uint64_t> num_splits;
    process_args(argc, argv, mode, enable_traverse, launch_mode, num_splits);

    init_distributed(rank, world_size);
    CUDA_CHECK(cudaSetDevice(rank));
    
    if (enable_traverse) {
        ep_moe_traverse(rank, world_size, mode, launch_mode, num_splits);
    } else {
        ep_moe(
            128/*num_experts*/, 
            1024/*num_max_dispatch_tokens_per_rank*/, 
            3072/*khidden*/, 
            4096/*hidden_size*/, 
            512/*num_tokens*/, 
            8/*num_topk*/, 
            world_size,
            mode,
            launch_mode,
            num_splits);
    }

    destroy_distributed();
    return 0;
}