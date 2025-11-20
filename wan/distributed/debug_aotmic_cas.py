import torch
import torch.distributed as dist
import os
import triton
import triton.language as tl
import iris
from triton_kernels import *


import aiter





# @triton.jit
# def load_data_from_target_rank(iris_input_buffer,
#                     hs:tl.constexpr,
#                     in_hn:tl.constexpr, # 5
#                     seq_this_rank:tl.constexpr, # 109120
#                     local_rank:tl.constexpr,
#                     world_size:tl.constexpr,
#                     local_output_bufer,
#                     target_rank:tl.constexpr,
#                     heap_bases:tl.tensor):
#     pid = tl.program_id(0)
#     output_seq_len = seq_this_rank // world_size
#     out_hn = in_hn * world_size
#     remote_data_start_offset = (local_rank * output_seq_len)*(in_hn * hs) + pid * (in_hn * hs)
#     local_output_start_offset = pid * out_hn * hs
#     for head_idx in tl.range(0,in_hn):
#         remote_ptrs = iris_input_buffer + remote_data_start_offset + head_idx * hs + tl.arange(0,hs)
#         head_data = iris.load(
#             pointer = remote_ptrs,
#             to_rank = local_rank,
#             from_rank = target_rank,
#             heap_bases = heap_bases,
#             mask = None
#         )
#         tl.store(
#             pointer=local_output_bufer + local_output_start_offset + target_rank * in_hn * hs + head_idx *hs + tl.arange(0,hs),
#             value = head_data,
#             mask = None
#         )


# @triton.jit
# def all_to_all_4D_bf16_backward_no_barrier_backup(iris_input_buffer,
#                     hs:tl.constexpr,
#                     in_hn:tl.constexpr, # 5
#                     seq_this_rank:tl.constexpr, # 109120
#                     local_rank:tl.constexpr,
#                     world_size:tl.constexpr,
#                     local_output_bufer,
#                     lock_base,lock_offset,
#                     heap_bases:tl.tensor):

#     pid = tl.program_id(0)
#     if pid == 0:
#         tl.store(lock_base + lock_offset,1,mask=None)
#     # iris.atomic_cas(pointer = lock_base + lock_offset, 
#     #                 cmp = 0, val = 1, 
#     #                 from_rank = local_rank, 
#     #                 to_rank = local_rank, 
#     #                 heap_bases=heap_bases)

#     finished_flags = 0
#     mask = (1 << world_size) - 1 # 1111 1111 
#     all_finished = ((finished_flags & mask) == mask)

#     # load local data first
#     load_data_from_target_rank(
#         iris_input_buffer,hs,in_hn,seq_this_rank,
#         local_rank,world_size,local_output_bufer,
#         local_rank,heap_bases)
#     finished_flags = finished_flags | (1 << local_rank)

#     while not all_finished:
#         for target_rank in tl.range(0,world_size,num_stages=2):
#             # check if this rank's data is loaded, if not, proceed
#             rank_finished = ((finished_flags >> target_rank) & 1) == 1
#             if not rank_finished:
#                 lock_released = (iris.atomic_cas(
#                     pointer = lock_base + lock_offset, cmp = 1, val = 1, 
#                     from_rank = local_rank, to_rank = target_rank, 
#                     heap_bases=heap_bases) == 1)
#                 if lock_released:
#                     load_data_from_target_rank(
#                         iris_input_buffer,hs,in_hn,seq_this_rank,
#                         local_rank,world_size,local_output_bufer,
#                         target_rank,heap_bases)
#                     finished_flags = finished_flags | (1 << target_rank)
#         all_finished = ((finished_flags & mask) == mask)



# @triton.jit
# def all_to_all_4D_bf16_backward_no_barrier(iris_input_buffer,
#                     hs:tl.constexpr,
#                     in_hn:tl.constexpr, # 5
#                     seq_this_rank:tl.constexpr, # 109120
#                     local_rank:tl.constexpr,
#                     world_size:tl.constexpr,
#                     local_output_bufer,
#                     lock_base,lock_offset,
#                     heap_bases:tl.tensor):

#     pid = tl.program_id(0)
#     # if pid == 0:
#     #     tl.store(lock_base + lock_offset,1,mask=None)
#     iris.atomic_cas(pointer = lock_base + lock_offset, 
#                     cmp = 0, val = 1, 
#                     from_rank = local_rank, 
#                     to_rank = local_rank, 
#                     heap_bases=heap_bases,
#                     sem = "release")

#     finished_flags = 0
#     mask = (1 << world_size) - 1 # 1111 1111 
#     all_finished = ((finished_flags & mask) == mask)

#     # load local data first
#     load_data_from_target_rank(
#         iris_input_buffer,hs,in_hn,seq_this_rank,
#         local_rank,world_size,local_output_bufer,
#         local_rank,heap_bases)
#     finished_flags = finished_flags | (1 << local_rank)

#     while not all_finished:
#         for target_rank in tl.range(0,world_size,num_stages=2):
#             # check if this rank's data is loaded, if not, proceed
#             rank_finished = ((finished_flags >> target_rank) & 1) == 1
#             if not rank_finished:
#                 lock_released = (iris.atomic_cas(
#                     pointer = lock_base + lock_offset, cmp = 1, val = 1, 
#                     from_rank = local_rank, to_rank = target_rank, 
#                     heap_bases=heap_bases,sem="acquire") == 1)
#                 if lock_released:
#                     load_data_from_target_rank(
#                         iris_input_buffer,hs,in_hn,seq_this_rank,
#                         local_rank,world_size,local_output_bufer,
#                         target_rank,heap_bases)
#                     finished_flags = finished_flags | (1 << target_rank)
#         all_finished = ((finished_flags & mask) == mask)






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
lock = iris_handle.zeros([1024],dtype=torch.int32,device="cuda")


q = torch.randn([1,13640,40,128],dtype=torch.bfloat16,device="cuda")
k = torch.randn([1,13640,40,128],dtype=torch.bfloat16,device="cuda")
v = torch.randn([1,13640,40,128],dtype=torch.bfloat16,device="cuda")
freqs_i = torch.randn([13640*8,128],dtype=torch.float64,device="cuda")

bs = q.shape[0]
hs = q.shape[-1]

sp_seq_len = q.shape[1]
hn = q.shape[2]
heap_bases = iris_handle.get_heap_bases()
seq_lens = 109120
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
                k_lens=torch.tensor([seq_lens]),
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


