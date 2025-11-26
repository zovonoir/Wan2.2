import torch
import torch.distributed as dist
import os
import triton
import triton.language as tl
import iris
from triton_kernels_gluon import *


import aiter


import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
import iris.experimental.iris_gluon as iris_gl


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



def flash_attention111(
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
    import flash_att
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
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

    x = flash_attn.flash_attn_varlen_func(
        # x = aiter.ops.mha.flash_attn_varlen_func(
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

# iris_handle = iris.iris(4*1024*1024*1024)
# iris_q = iris_handle.zeros([1,13640*8,40//8,128],dtype=torch.bfloat16,device="cuda")
# iris_k = iris_handle.zeros([1,13640*8,40//8,128],dtype=torch.bfloat16,device="cuda")
# iris_v = iris_handle.zeros([1,13640*8,40//8,128],dtype=torch.bfloat16,device="cuda")
# iris_o = iris_handle.zeros([1,13640*8,40//8,128],dtype=torch.bfloat16,device="cuda")
# attn_buffer = iris_handle.zeros([1,13640,40,128],dtype=torch.bfloat16,device="cuda")
# lock = iris_handle.zeros([1024],dtype=torch.int32,device="cuda")

ctx = iris_gl.iris(heap_size=2**30)
context_tensor = ctx.get_device_context()
iris_q = ctx.zeros([1,13640*8,40//8,128],dtype=torch.bfloat16,device="cuda")
iris_k = ctx.zeros([1,13640*8,40//8,128],dtype=torch.bfloat16,device="cuda")
iris_v = ctx.zeros([1,13640*8,40//8,128],dtype=torch.bfloat16,device="cuda")
iris_o = ctx.zeros([1,13640*8,40//8,128],dtype=torch.bfloat16,device="cuda")
attn_buffer = ctx.zeros([1,13640,40,128],dtype=torch.bfloat16,device="cuda")
lock = ctx.zeros([1024],dtype=torch.int32,device="cuda")
q = torch.randn([1,13640,40,128],dtype=torch.bfloat16,device="cuda")
k = torch.randn([1,13640,40,128],dtype=torch.bfloat16,device="cuda")
v = torch.randn([1,13640,40,128],dtype=torch.bfloat16,device="cuda")

# attn_buffer2 = torch.zeros([1,13640,40,128],dtype=torch.bfloat16,device="cuda")

freqs_i = torch.randn([13640*8,128],dtype=torch.float64,device="cuda")

bs = q.shape[0]
hs = q.shape[-1]

sp_seq_len = q.shape[1]
hn = q.shape[2]
# heap_bases = iris_handle.get_heap_bases()
seq_lens = 109120
window_size = (-1,-1)






# warmup
for _ in range(50):
    break
    lock[0] += 1
    lock[lock[0]] = 0
    # iris_handle.barrier()
    ctx.barrier()
    # all_to_all_4D_bf16_backward[(13640,1,1)](iris_o,
    #                         hs,
    #                         hn//world_size, # 5
    #                         sp_seq_len*world_size, # 109120
    #                         rank,
    #                         8,
    #                         attn_buffer,
    #                         heap_bases)
    alltoallbackward[(608,1,1)](
                    iris_gl.IrisDeviceCtx,
            context_tensor,
        iris_input_buffer = iris_o,
                                    hs = hs,
                                    in_hn = hn//world_size,
                                    seq_this_rank = sp_seq_len*world_size,
                                    local_rank = rank,
                                    world_size = world_size,
                                    local_output_bufer = attn_buffer,
                                    lock_base = lock,
                                    lock_offset = lock[0].item())

# iris_handle.barrier()
ctx.barrier()



start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

total_time = 0
n=200
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=False,
    with_stack=True,
    with_flops=False
) as prof:
    for _ in range(n):
        # rope_alltoall_4D_bf16_forward[(sp_seq_len, 1, 1)](q, freqs_i, hs, rank, sp_seq_len, hn, iris_q, world_size, 1)
        # rope_alltoall_4D_bf16_forward[(sp_seq_len, 1, 1)](k, freqs_i, hs, rank, sp_seq_len, hn, iris_k, world_size, 1)
        # all_to_all_4D_bf16_forward[(sp_seq_len, 1, 1)](v, hs, hn, sp_seq_len, rank, world_size, iris_v, 1)
        lock[0] += 1
        lock[lock[0]] = 0
        # iris_handle.barrier()
        ctx.barrier()

        iris_o.copy_(
            flash_attention(
                iris_q,
                iris_k,
                iris_v,
                k_lens=torch.tensor([seq_lens]),
                window_size=window_size,
            )
        )
        # iris_handle.barrier()
        ctx.barrier()
        
        alltoallbackward[(304*2,1,1)](
                        iris_gl.IrisDeviceCtx,
            context_tensor,
            iris_input_buffer = iris_o,
                                        hs = hs,
                                        in_hn = hn//world_size,
                                        seq_this_rank = sp_seq_len*world_size,
                                        local_rank = rank,
                                        world_size = world_size,
                                        local_output_bufer = attn_buffer,
                                        lock_base = lock,
                                        lock_offset = lock[0].item())

        # all_to_all_4D_bf16_backward[(13640,1,1)](iris_o,
        #                     hs,
        #                     hn//world_size, # 5
        #                     sp_seq_len*world_size, # 109120
        #                     rank,
        #                     8,
        #                     attn_buffer,
        #                     heap_bases)



        # end_event.record()
        # torch.cuda.synchronize()
        # elapsed_time = start_event.elapsed_time(end_event) #ms
        # elapsed_time_sec = elapsed_time / 1000.0 #seconds
        # total_time += elapsed_time

print(f"rank {rank} average time(ms):{total_time / n}")

trace_path = f"trace_rank{rank}.json"
prof.export_chrome_trace(trace_path)
print(f"[rank{rank}] trace saved to {trace_path}")

dist.destroy_process_group()


