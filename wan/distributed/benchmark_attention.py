
# import flash_attn

import torch
import os
import torch.distributed as dist
import aiter 


import argparse


parser = argparse.ArgumentParser(description='Flash Attention Forward Performance Test')
    
# Tensor shape and attention parameters
parser.add_argument('-rank', type=int, required=True, help='Batch size for input tensors')

args  = parser.parse_args()


# 设置单线程，避免CPU竞争
torch.set_num_threads(1)

# rank = int(os.environ["RANK"])
# world_size =  int(os.environ["WORLD_SIZE"])
# dist.init_process_group(
#     backend="nccl",
#     device_id=torch.device(f"cuda:{rank}"),
#     world_size=world_size)

rank = args.rank
world_size = 8

# 设置当前进程使用的GPU
torch.cuda.set_device(rank)
device = f"cuda:{rank}"

print(f"Rank {rank}: Using device {device}, CPU affinity: {os.sched_getaffinity(0)}")

# 在指定的GPU上创建数据
iris_q = torch.randn([1,13640*8,5,128],dtype=torch.bfloat16,device=device)
iris_k = torch.randn([1,13640*8,5,128],dtype=torch.bfloat16,device=device)
iris_v = torch.randn([1,13640*8,5,128],dtype=torch.bfloat16,device=device)

garbage = []

# for i in range(25): # 50G
#     garbage.append(torch.empty([1024,1024,1024],dtype=torch.half,device="cuda"))

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

    if q_scale is not None:
        q = q * q_scale
    # apply attention
    x = flash_attn.flash_attn_varlen_func(
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

def flash_attention_aiter(
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

    if q_scale is not None:
        q = q * q_scale
    # apply attention
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

import random

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
    record_shapes=False,
    profile_memory=False,
    with_stack=True,
    with_flops=False
) as prof:
    for i in range(200):
        flash_attention_aiter(
            iris_q,
            iris_k,
            iris_v,
            k_lens=torch.tensor([109120],dtype=torch.long,device=device),
            window_size=(-1,-1),
        )
trace_path = f"flash_attention_pressure_testing_rank_{rank}_world_size_{world_size}.json"
prof.export_chrome_trace(trace_path)
print(f"trace saved to {trace_path}")


# for i in range(5):
#     flash_attention_aiter(
#         iris_q,
#         iris_k,
#         iris_v,
#         k_lens=torch.tensor([109120],dtype=torch.long,device=device),
#         window_size=(-1,-1),
#     )

torch.cuda.synchronize()



