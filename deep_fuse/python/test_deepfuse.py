import torch
import torch.distributed as dist
import os
import torch.multiprocessing as mp
import deep_fuse
import argparse
import numpy as np

NUM_EXPERTS, NUM_MAX_DISPATCH_TOKENS_PER_RANK, KHIDDEN, HIDDEN_SIZE, NUM_TOKENS, NUM_TOPK = 128, 1024, 3072, 4096, 512, 8

MODE = "sequence" # "sequence" or "multi_token"
LAUNCH_MODE = "sched" # "sched" or "sync"
SMS = 20

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

def test_deepfuse(rank, world_size, args):
    local_rank = rank
    rank = world_size * args.node_rank + rank
    world_size = world_size * args.nnodes
    dist.init_process_group("nccl", init_method=f"tcp://{args.master_ip}:{args.master_port}", rank=rank, world_size=world_size)
    backend = dist.group.WORLD._get_backend(torch.device("cuda"))
    NUM_EXPERTS, NUM_MAX_DISPATCH_TOKENS_PER_RANK, KHIDDEN, HIDDEN_SIZE, NUM_TOKENS, NUM_TOPK = \
        args.num_experts, args.num_max_dispatch_tokens_per_rank, args.khidden, args.hidden_size, args.num_tokens, args.num_topk
    MODE, LAUNCH_MODE, SMS = \
        args.mode, args.launch_mode, args.sms
    torch.cuda.set_device(local_rank)
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
    runtime.create_mode(num_splits=2)
    runtime.load_weights(w13_weight=w13_weight_fp8, w2_weight=w2_weight_fp8)

    for i in range(50):
        topk_idx = torch.randint(0, NUM_EXPERTS, (NUM_TOKENS, NUM_TOPK), dtype=torch.int64, device="cuda")
        topk_weights = torch.randn(NUM_TOKENS, NUM_TOPK, dtype=torch.float32, device="cuda")
        hidden_states = torch.randn(NUM_TOKENS, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
        runtime.get_metadata(mode=MODE, num_tokens=NUM_TOKENS, num_split_tokens=[])
        runtime.load_inputs(mode=MODE, hidden_states_in=hidden_states, topk_ids_in=topk_idx, topk_weights_in=topk_weights)
        runtime.launch(mode=MODE, launch_mode=LAUNCH_MODE, deepep_sms=SMS)
        final_hidden_states = runtime.get_merged_output(mode=MODE)
        
    start_events = []
    end_events = []
    for i in range(20):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        topk_idx = torch.randint(0, NUM_EXPERTS, (NUM_TOKENS, NUM_TOPK), dtype=torch.int64, device="cuda")
        topk_weights = torch.randn(NUM_TOKENS, NUM_TOPK, dtype=torch.float32, device="cuda")
        hidden_states = torch.randn(NUM_TOKENS, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
        runtime.get_metadata(mode=MODE, num_tokens=NUM_TOKENS, num_split_tokens=[])
        runtime.load_inputs(mode=MODE, hidden_states_in=hidden_states, topk_ids_in=topk_idx, topk_weights_in=topk_weights)
        start_event.record()
        runtime.launch(mode=MODE, launch_mode=LAUNCH_MODE, deepep_sms=SMS)
        end_event.record()
        final_hidden_states = runtime.get_merged_output(mode=MODE)
        start_events.append(start_event)
        end_events.append(end_event)
    
    torch.distributed.barrier()   
    torch.cuda.synchronize()
    elapsed_times = []
    for i in range(20):
        elapsed_times.append(start_events[i].elapsed_time(end_events[i]))
        
    if rank == 0:
        print(f"World Size [{world_size}] Batch Size[{NUM_TOKENS}] Mode[{MODE}] Launch Mode[{LAUNCH_MODE}] SMS[{SMS}]: Average time: {np.mean(elapsed_times)} ms")

    torch.distributed.barrier()
    dist.destroy_process_group()
    
def test_comm_interface(rank, world_size, args):
    rank = world_size * args.node_rank + rank
    world_size = world_size * args.nnodes
    dist.init_process_group("nccl", init_method=f"tcp://{args.master_ip}:{args.master_port}", rank=rank, world_size=world_size)
    backend = dist.group.WORLD._get_backend(torch.device("cuda"))
    NUM_EXPERTS, NUM_MAX_DISPATCH_TOKENS_PER_RANK, KHIDDEN, HIDDEN_SIZE, NUM_TOKENS, NUM_TOPK = \
        args.num_experts, args.num_max_dispatch_tokens_per_rank, args.khidden, args.hidden_size, args.num_tokens, args.num_topk
    MODE, LAUNCH_MODE, SMS = \
        args.mode, args.launch_mode, args.sms
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
    runtime.create_mode(num_splits=4)
    runtime.load_weights(w13_weight=w13_weight_fp8, w2_weight=w2_weight_fp8)

    for i in range(50):
        topk_idx = torch.randint(0, NUM_EXPERTS, (NUM_TOKENS, NUM_TOPK), dtype=torch.int64, device="cuda")
        topk_weights = torch.randn(NUM_TOKENS, NUM_TOPK, dtype=torch.float32, device="cuda")
        hidden_states = torch.randn(NUM_TOKENS, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")

        final_hidden_states = torch.randn(NUM_EXPERTS // world_size, NUM_MAX_DISPATCH_TOKENS_PER_RANK * world_size, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
        (
            hidden_states,
            masked_m,
            expected_m
        ) = runtime.low_latency_dispatch_interface(
            mode="sequence",
            deepep_sms=20
        )
        final_hidden_states = runtime.low_latency_combine_interface(
            mode="sequence",
            compute_result=final_hidden_states,
            deepep_sms=20
        )

    torch.distributed.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    os.environ["NCCL_DEBUG"] = "WARN"
    world_size = 8
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--master-ip", type=str, default="localhost")
    parser.add_argument("--master-port", type=int, default=29500)
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--node-rank", type=int, default=0)
    parser.add_argument("--num-experts", type=int, default=128)
    parser.add_argument("--num-max-dispatch-tokens-per-rank", type=int, default=1024)
    parser.add_argument("--khidden", type=int, default=3072)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--num-tokens", type=int, default=512)
    parser.add_argument("--num-topk", type=int, default=8)
    
    parser.add_argument("--mode", type=str, default="multi_token")
    parser.add_argument("--launch-mode", type=str, default="sched")
    parser.add_argument("--sms", type=int, default=20)
    
    args = parser.parse_args()
    
    world_size = int(args.world_size / args.nnodes)
    
    mp.spawn(test_deepfuse, args=(world_size, args,), nprocs=world_size, join=True)