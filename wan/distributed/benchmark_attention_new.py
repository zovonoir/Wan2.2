import torch
import os
import torch.distributed as dist
import aiter 

rank = int(os.getenv("RANK", 0))
world_size = int(os.getenv("WORLD_SIZE", 1))
local_rank = int(os.getenv("LOCAL_RANK", 0))
device = local_rank


torch.cuda.set_device(local_rank)
dist.init_process_group(
    backend="nccl",
    init_method="env://",
    rank=rank,
    world_size=world_size)


