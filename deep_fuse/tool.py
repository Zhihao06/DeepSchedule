import torch
import torch.distributed as dist
import deep_fuse_cpp
from typing import Callable, List, Tuple, Optional, Union

class Tool:
    def __init__(self, num_experts: int, num_max_dispatch_tokens_per_rank: int, khidden: int, hidden_size: int, num_tokens: int, 
            num_topk: int, world_size: int, global_pg_nccl: dist.ProcessGroupNCCL) -> None:
        self.num_experts = num_experts
        self.num_max_dispatch_tokens_per_rank = num_max_dispatch_tokens_per_rank
        self.khidden = khidden
        self.hidden_size = hidden_size
        self.num_tokens = num_tokens
        self.num_topk = num_topk
        self.world_size = world_size
        self.global_pg_nccl = global_pg_nccl
        self.runtime = deep_fuse_cpp.Tool(self.num_experts, self.num_max_dispatch_tokens_per_rank, self.khidden, self.hidden_size, self.num_tokens, self.num_topk, self.world_size, self.global_pg_nccl)
    
    def create_mode(self, num_splits: int) -> None:
        self.runtime.create_mode(num_splits)

    def get_metadata(self, mode: str, num_tokens: int, num_split_tokens: List[int]) -> None:
        self.runtime.get_metadata(mode, num_tokens, num_split_tokens)

    def load_inputs(self, mode: str, hidden_states_in: torch.Tensor, topk_ids_in: torch.Tensor, topk_weights_in: torch.Tensor) -> None:
        self.runtime.load_inputs(mode, hidden_states_in, topk_ids_in, topk_weights_in)

    def load_weights(self, w13_weight: Tuple[torch.Tensor, torch.Tensor], w2_weight: Tuple[torch.Tensor, torch.Tensor]) -> None:
        self.runtime.load_weights(w13_weight[0], w13_weight[1], w2_weight[0], w2_weight[1])

    def launch(self, mode: str, launch_mode: str, deepep_sms: int) -> None:
        self.runtime.launch(mode, launch_mode, deepep_sms)

    def get_merged_output(self, mode: str) -> torch.Tensor:
        return self.runtime.get_merged_output(mode)

    def low_latency_dispatch_interface(self, mode: str, deepep_sms: int) -> Tuple[Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor], torch.Tensor, int]:
        packed_recv_x, packed_recv_x_scales, packed_recv_count, expected_m = self.runtime.low_latency_dispatch_interface(mode, deepep_sms)
        return (packed_recv_x, packed_recv_x_scales) if packed_recv_x_scales is not None else packed_recv_x, packed_recv_count, expected_m

    def low_latency_combine_interface(self, mode: str, deepep_sms: int) -> torch.Tensor:
        return self.runtime.low_latency_combine_interface(mode, deepep_sms)
