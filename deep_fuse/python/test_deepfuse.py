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
        num_experts=NUM_EXPERTS, 
        num_max_dispatch_tokens_per_rank=NUM_MAX_DISPATCH_TOKENS_PER_RANK, 
        khidden=KHIDDEN, 
        hidden_size=HIDDEN_SIZE, 
        num_tokens=NUM_TOKENS, 
        num_topk=NUM_TOPK, 
        world_size=world_size, 
        global_pg_nccl=dist.group.WORLD._get_backend(torch.device("cuda"))
    )
    w13_weight_fp8, w2_weight_fp8 = get_expert_weights()
    topk_idx = torch.randint(0, NUM_EXPERTS, (NUM_TOKENS, NUM_TOPK), dtype=torch.int64, device="cuda")
    topk_weights = torch.randn(NUM_TOKENS, NUM_TOPK, dtype=torch.float32, device="cuda")
    hidden_states = torch.randn(NUM_TOKENS, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")

    runtime.create_mode(num_splits=4)
    runtime.load_weights(w13_weight=w13_weight_fp8, w2_weight=w2_weight_fp8)
    runtime.get_metadata(mode="multi_token", num_tokens=512, num_split_tokens=[128, 128, 128, 128])
    runtime.load_inputs(mode="multi_token", hidden_states_in=hidden_states, topk_ids_in=topk_idx, topk_weights_in=topk_weights)
    runtime.launch(mode="multi_token", launch_mode="sched", deepep_sms=20)
    final_hidden_states = runtime.get_merged_output(mode="multi_token")

    torch.distributed.barrier()
    dist.destroy_process_group()
    

if __name__ == "__main__":
    os.environ["NCCL_DEBUG"] = "WARN"
    world_size = 8
    mp.spawn(test_deepfuse, args=(world_size,), nprocs=world_size, join=True)