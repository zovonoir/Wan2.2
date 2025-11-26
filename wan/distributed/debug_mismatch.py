import torch
import torch.distributed as dist
import os
import triton
import triton.language as tl
import iris

import aiter

from triton_kernels import *


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda' and q.size(-1) <= 256

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor(
            [lq] * b, dtype=torch.int32).to(
                device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor(
            [lk] * b, dtype=torch.int32).to(
                device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )

    x = aiter.ops.mha.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic).unflatten(0, (b, lq))

    # output
    return x.type(out_dtype)



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
lock = iris_handle.zeros([4096],dtype=torch.int32,device="cuda")


q = torch.randn([1,13640,40,128],dtype=torch.bfloat16,device="cuda")
k = torch.randn([1,13640,40,128],dtype=torch.bfloat16,device="cuda")
v = torch.randn([1,13640,40,128],dtype=torch.bfloat16,device="cuda")

# attn_buffer2 = torch.zeros([1,13640,40,128],dtype=torch.bfloat16,device="cuda")

freqs_i = torch.randn([13640*8,128],dtype=torch.float64,device="cuda")

bs = q.shape[0]
hs = q.shape[-1]

sp_seq_len = q.shape[1]
hn = q.shape[2]
heap_bases = iris_handle.get_heap_bases()
seq_lens = 109120
window_size = (-1,-1)


# warmup
for _ in range(10):
    all_to_all_4D_bf16_backward_new[(sp_seq_len,1,1)](iris_o,128,5, sp_seq_len*world_size,rank, world_size, attn_buffer, heap_bases)

iris_handle.barrier()

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

total_time = 0
n=100
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=False,
    with_stack=True,
    with_flops=False
) as prof:
    for _ in range(n):
        rope_alltoall_4D_bf16_forward[(sp_seq_len, 1, 1)](q, freqs_i, hs, rank, sp_seq_len, hn, iris_q, world_size, heap_bases)
        rope_alltoall_4D_bf16_forward[(sp_seq_len, 1, 1)](k, freqs_i, hs, rank, sp_seq_len, hn, iris_k, world_size, heap_bases)
        all_to_all_4D_bf16_forward[(sp_seq_len, 1, 1)](v, hs, hn, sp_seq_len, rank, world_size, iris_v, heap_bases)
        iris_handle.barrier()

        _o = flash_attention(
                iris_q,
                iris_k,
                iris_v,
                k_lens=torch.tensor([seq_lens]),
                window_size=(-1,-1),
            )
        all_to_all_4D_bf16_backward_new[(sp_seq_len,1,1)](_o,128,5, sp_seq_len*world_size,rank, world_size, attn_buffer, heap_bases)
        iris_handle.barrier()


print(f"rank {rank} average time(ms):{total_time / n}")

trace_path = f"trace_rank{rank}.json"
prof.export_chrome_trace(trace_path)
print(f"[rank{rank}] trace saved to {trace_path}")

dist.destroy_process_group()


