import torch
import torch.distributed as dist
import os
import triton
import triton.language as tl
from triton_kernels import *
from ..modules.attention import flash_attention

def get_rank():
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    return local_rank
rank = get_rank()

def get_world_size():
    world_size = int(os.getenv("WORLD_SIZE", 1))
    return world_size

world_size = get_world_size()
local_rank = int(os.getenv("LOCAL_RANK", 0))
device = local_rank


torch.cuda.set_device(local_rank)
dist.init_process_group(
    backend="nccl",
    init_method="env://",
    rank=rank,
    world_size=world_size)

iris_handle = iris.iris(4*1024*1024*1024)
iris_q = iris_handle.zeros([1,13640*8,40//8,128],dtype=torch.bfloat16,device="cuda")
iris_k = iris_handle.zeros([1,13640*8,40//8,128],dtype=torch.bfloat16,device="cuda")
iris_v = iris_handle.zeros([1,13640*8,40//8,128],dtype=torch.bfloat16,device="cuda")
iris_o = iris_handle.zeros([1,13640*8,40//8,128],dtype=torch.bfloat16,device="cuda")
attn_buffer = iris_handle.zeros([1,13640,40,128],dtype=torch.bfloat16,device="cuda")
lock = iris_handle.zeros([1024],dtype=torch.int32)


q = torch.randn([1,13640,40,128],dtype=torch.bfloat16,device="cuda")
k = torch.randn([1,13640,40,128],dtype=torch.bfloat16,device="cuda")
v = torch.randn([1,13640,40,128],dtype=torch.bfloat16,device="cuda")
freqs_i = torch.randn([1024,64],dtype=torch.float32,device="cuda")

bs = q.shape[0]
hs = q.shape[-1]

sp_seq_len = q.shape[1]
hn = q.shape[2]
heap_bases = iris_handle.get_heap_bases()
seq_lens = 13640
window_size = (-1,-1)

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=False,
    with_stack=True,
    with_flops=False
) as prof:
    for _ in range(50):
        rope_alltoall_4D_bf16_forward[(sp_seq_len, 1, 1)](q, freqs_i, hs, rank, sp_seq_len, hn, iris_q, world_size, heap_bases)
        rope_alltoall_4D_bf16_forward[(sp_seq_len, 1, 1)](k, freqs_i, hs, rank, sp_seq_len, hn, iris_k, world_size, heap_bases)
        all_to_all_4D_bf16_forward[(sp_seq_len, 1, 1)](v, hs, hn, sp_seq_len, rank, world_size, iris_v, heap_bases)
        lock[0] += 1
        lock[lock[0]] = 0
        iris_handle.barrier()

        iris_o.copy_(
            flash_attention(
                iris_q,
                iris_k,
                iris_v,
                k_lens=seq_lens,
                window_size=window_size,
            )
        )

        all_to_all_4D_bf16_backward_no_barrier[(13640,1,1)](
                                        iris_input_buffer = iris_o,
                                        hs = hs,
                                        in_hn = hn//world_size,
                                        seq_this_rank = sp_seq_len*world_size,
                                        local_rank = rank,
                                        world_size = world_size,
                                        local_output_bufer = attn_buffer,
                                        lock_base = lock,
                                        lock_offset = lock[0].item(),
                                        heap_bases = heap_bases)

trace_path = f"trace_rank{rank}.json"
prof.export_chrome_trace(trace_path)
print(f"[rank{rank}] trace saved to {trace_path}")

dist.destroy_process_group()


