import torch
import torch.distributed as dist
import os
import torch.multiprocessing as mp
import deep_fuse

NUM_EXPERTS, NUM_MAX_DISPATCH_TOKENS_PER_RANK, KHIDDEN, HIDDEN_SIZE, NUM_TOKENS, NUM_TOPK = 128, 1024, 3072, 4096, 512, 8

def get_expert_weights():
    num_group = NUM_EXPERTS // dist.get_world_size()
    w13_weight_fp8 = (
        torch.randn(num_group, KHIDDEN, HIDDEN_SIZE, dtype=torch.float32, device="cuda").to(torch.float8_e4m3fn),
        torch.randn(num_group, (KHIDDEN + 127) // 128, HIDDEN_SIZE // 128, dtype=torch.float32, device="cuda")
    )
    w2_weight_fp8 = (
        torch.randn(num_group, HIDDEN_SIZE, KHIDDEN // 2, dtype=torch.float32, device="cuda").to(torch.float8_e4m3fn), 
        torch.randn(num_group, (HIDDEN_SIZE + 127) // 128, KHIDDEN // 2 // 128, dtype=torch.float32, device="cuda")
    )
    return w13_weight_fp8, w2_weight_fp8

def test_deepfuse(rank, world_size):
    dist.init_process_group("nccl", init_method="tcp://127.0.0.1:29500", rank=rank, world_size=world_size)
    backend = dist.group.WORLD._get_backend(torch.device("cuda"))
    print("backend rank", backend.rank())
    torch.cuda.set_device(backend.rank())
    runtime = deep_fuse.Tool(
        NUM_EXPERTS, NUM_MAX_DISPATCH_TOKENS_PER_RANK, KHIDDEN, HIDDEN_SIZE, NUM_TOKENS, NUM_TOPK, 
        world_size, dist.group.WORLD._get_backend(torch.device("cuda"))
    )
    w13_weight_fp8, w2_weight_fp8 = get_expert_weights()
    topk_idx = torch.randint(0, NUM_EXPERTS, (NUM_TOKENS, NUM_TOPK), dtype=torch.int64, device="cuda")
    topk_weights = torch.randn(NUM_TOKENS, NUM_TOPK, dtype=torch.float32, device="cuda")
    hidden_states = torch.randn(NUM_TOKENS, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")

    runtime.create_mode(4) # num_splits
    runtime.load_weights(w13_weight_fp8, w2_weight_fp8)
    runtime.get_metadata("multi_token", 512, [128, 128, 128, 128])
    runtime.load_inputs("multi_token", hidden_states, topk_idx, topk_weights)
    runtime.launch("multi_token", "sched", 20)
    final_hidden_states = runtime.get_merged_output("multi_token")

    torch.distributed.barrier()
    dist.destroy_process_group()
    

if __name__ == "__main__":
    os.environ["NCCL_DEBUG"] = "WARN"
    world_size = 8
    mp.spawn(test_deepfuse, args=(world_size,), nprocs=world_size, join=True)