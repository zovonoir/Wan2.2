from numpy import dtype
import torch
import os
import torch.distributed as dist


rank = int(os.environ["RANK"])
world_size =  int(os.environ["WORLD_SIZE"])
dist.init_process_group(
    backend="nccl",
    device_id=torch.device(f"cuda:{rank}"),
    world_size=world_size)


garbage = []
for i in range(10):
    garbage.append(torch.randn([1024,1024,1024],dtype=torch.half,device="cuda"))
if rank == 0:
    os.system("rocm-smi")
    os.system("amd-smi")
print(len(garbage))
