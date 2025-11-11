import os
import torch
import torch.distributed as dist
import iris
rank = int(os.environ["RANK"])
world_size =  int(os.environ["WORLD_SIZE"])
dist.init_process_group(
    backend="nccl",
    device_id=torch.device(f"cuda:{rank}"),
    world_size=world_size,
    rank=rank,
    init_method="tcp://127.0.0.1:29500")
shmem = iris.iris(13640*40*128*2)
iris_o = shmem.zeros([1,13640*8,40//8,128],dtype=torch.bfloat16,device="cuda")

import triton
import triton.language as tl



@triton.jit
def rope_triton_kernel_fp16_with_alltoall(qk_ptr, freqs_ptr, 
                    # output_ptr,
                    PROG_SIZE:tl.constexpr,# = head_size
                    sp_rank:tl.constexpr, # [0-7]
                    s_per_rank:tl.constexpr, # 13640
                    head_num:tl.constexpr, # 40
                    # alltoall
                    iris_buffer,
                    world_size:tl.constexpr,
                    heap_bases:tl.tensor
                    ):
    # tl.static_assert(0)
    # 每个program负责一个token,共40*128=5120个float32相乘
    # 但是freq只有128个,要broadcast到[40,128]
    # PROG_SIZE 必须等于head_size!
    program_id = tl.program_id(0)
    tl.static_assert(PROG_SIZE > 0 and (PROG_SIZE & (PROG_SIZE - 1)) == 0,f"PROG_SIZE only support power of 2!current is {PROG_SIZE}")
    freqs_rank_offset = PROG_SIZE * sp_rank * s_per_rank # freq当前rank在freq中的偏移量
    freq_rank_start_ptr = freqs_ptr + freqs_rank_offset # 这个rank的起始指针
    freq_program_start_ptr = freq_rank_start_ptr + program_id * PROG_SIZE
    # program_start = program_id * PROG_SIZE
    # rank_process_size = PROG_SIZE * s_per_rank * head_num # 这个rank一共要处理这么多的数据
    even_offset = 2 * (tl.arange(0,PROG_SIZE) // 2) # program_start + [0,0,2,2,4,4,...126,126]
    odd_offset = even_offset + 1 # program_start + [1,1,3,3,5,5,...,127,127]
    freqs_offset = tl.arange(0,PROG_SIZE) #  [0,1,2,3,4,5...],freq只能load128个数据,并且复用
    freqs_offset_swap = tl.arange(0,PROG_SIZE) ^ 0x0001 # [1,0,3,2,5,4...]
    coe = 2*(tl.arange(0,PROG_SIZE) % 2) - 1 # [-1,1,-1,1,-1,1...]
    even_mask = even_offset < PROG_SIZE # all true
    odd_mask = odd_offset < PROG_SIZE
    freqs_mask = freqs_offset < PROG_SIZE
    freqs_mask_swap = freqs_offset_swap < PROG_SIZE
    vy1_fp64 = tl.cast(tl.load(freq_program_start_ptr + freqs_offset,mask=freqs_mask,other=0.0),tl.float64)
    vy2_fp64 = tl.cast(tl.load(freq_program_start_ptr + freqs_offset_swap,mask=freqs_mask_swap,other=0.0),tl.float64)
    vx1_ptr_block_even = qk_ptr + program_id * PROG_SIZE * head_num + even_offset # 当前BLOCK处理数据的第0个head的起始地址 + 每个数据偏移量
    vx1_ptr_block_odd = qk_ptr + program_id * PROG_SIZE * head_num + odd_offset
    # output_ptr_block = output_ptr + program_id * PROG_SIZE * head_num + tl.arange(0,PROG_SIZE)
    output_mask = tl.arange(0,PROG_SIZE) < PROG_SIZE
    # alltoall 4D输出形状
    input_head_num = head_num
    output_head_num = input_head_num // world_size
    input_seq_len = s_per_rank
    output_seq_len = input_seq_len * world_size
    # [1,input_seq_len,input_head_num,head_size] -> [1,input_seq_len * world_size,input_head_num / world_size,head_size]

    for head_idx in tl.range(0,head_num,1): # 在head 维度循环,每个program计算一个token的数据量
        # load single head for x
        vx1_fp64 = tl.cast(tl.load(vx1_ptr_block_even + head_idx * PROG_SIZE,mask=even_mask,other=0.0),tl.float64) # mask可能不严谨
        vx2_fp64 = tl.cast(tl.load(vx1_ptr_block_odd + head_idx * PROG_SIZE,mask=odd_mask,other=0.0),tl.float64)
        # complex_output = tl.cast(vx1*vy1 + coe*vx2*vy2,tl.float32) # 一个head的输出
        complex_output_bf16 = tl.cast(vx1_fp64*vy1_fp64 + coe*vx2_fp64*vy2_fp64,tl.bfloat16) # rope输出是float32格式,但是后续做alltoall之前又被转换成了bfloat16,因此这里提前转换再写入
        # tl.store(output_ptr_block + head_idx * PROG_SIZE,complex_output,mask=output_mask)
        # 这个head数据需要写到哪张卡哪个地址上?

        target_rank = head_idx // output_head_num
        head_idx_in_target_rank = head_idx - ((head_idx // output_head_num) * output_head_num) # 0-4 per rank
        token_id_in_target_rank = sp_rank * input_seq_len + program_id # 0-109120
        offset_in_target_rank = output_head_num * PROG_SIZE * token_id_in_target_rank + PROG_SIZE * head_idx_in_target_rank
        target_pointers = iris_buffer + offset_in_target_rank + tl.arange(0,PROG_SIZE)
        iris.store(
                pointer = target_pointers, # iris_buffer + head_idx*PROG_SIZE +tl.arange(0,PROG_SIZE), #target_pointers,
                value = complex_output_bf16,
                from_rank = sp_rank,
                to_rank = target_rank,
                heap_bases = heap_bases,
                mask = None
            )

@triton.jit
def rope_triton_kernel_fp16(qk_ptr, freqs_ptr, 
                    output_ptr,
                    PROG_SIZE:tl.constexpr,# = head_size
                    sp_rank:tl.constexpr, # [0-7]
                    s_per_rank:tl.constexpr, # 13640
                    head_num:tl.constexpr
                    ):
    program_id = tl.program_id(0)
    tl.static_assert(PROG_SIZE > 0 and (PROG_SIZE & (PROG_SIZE - 1)) == 0,f"PROG_SIZE only support power of 2!current is {PROG_SIZE}")
    freqs_rank_offset = PROG_SIZE * sp_rank * s_per_rank # freq当前rank在freq中的偏移量
    freq_rank_start_ptr = freqs_ptr + freqs_rank_offset # 这个rank的起始指针
    freq_program_start_ptr = freq_rank_start_ptr + program_id * PROG_SIZE
    even_offset = 2 * (tl.arange(0,PROG_SIZE) // 2) # program_start + [0,0,2,2,4,4,...126,126]
    odd_offset = even_offset + 1 # program_start + [1,1,3,3,5,5,...,127,127]
    freqs_offset = tl.arange(0,PROG_SIZE) #  [0,1,2,3,4,5...],freq只能load128个数据,并且复用
    freqs_offset_swap = tl.arange(0,PROG_SIZE) ^ 0x0001 # [1,0,3,2,5,4...]
    coe = 2*(tl.arange(0,PROG_SIZE) % 2) - 1 # [-1,1,-1,1,-1,1...]
    even_mask = even_offset < PROG_SIZE # all true
    odd_mask = odd_offset < PROG_SIZE
    freqs_mask = freqs_offset < PROG_SIZE
    freqs_mask_swap = freqs_offset_swap < PROG_SIZE
    vy1_fp64 = tl.cast(tl.load(freq_program_start_ptr + freqs_offset,mask=freqs_mask,other=0.0),tl.float64)
    vy2_fp64 = tl.cast(tl.load(freq_program_start_ptr + freqs_offset_swap,mask=freqs_mask_swap,other=0.0),tl.float64)
    vx1_ptr_block_even = qk_ptr + program_id * PROG_SIZE * head_num + even_offset # 当前BLOCK处理数据的第0个head的起始地址 + 每个数据偏移量
    vx1_ptr_block_odd = qk_ptr + program_id * PROG_SIZE * head_num + odd_offset
    output_ptr_block = output_ptr + program_id * PROG_SIZE * head_num + tl.arange(0,PROG_SIZE)
    output_mask = tl.arange(0,PROG_SIZE) < PROG_SIZE

    for head_idx in tl.range(0,head_num,1): # 在head 维度循环,每个program计算一个token的数据量
        # load single head for x
        vx1_fp64 = tl.cast(tl.load(vx1_ptr_block_even + head_idx * PROG_SIZE,mask=even_mask,other=0.0),tl.float64) # mask可能不严谨
        vx2_fp64 = tl.cast(tl.load(vx1_ptr_block_odd + head_idx * PROG_SIZE,mask=odd_mask,other=0.0),tl.float64)
        # complex_output = tl.cast(vx1*vy1 + coe*vx2*vy2,tl.float32) # 一个head的输出
        complex_output_bf16 = tl.cast(vx1_fp64*vy1_fp64 + coe*vx2_fp64*vy2_fp64,tl.bfloat16) # rope输出是float32格式,但是后续做alltoall之前又被转换成了bfloat16,因此这里提前转换再写入
        tl.store(output_ptr_block + head_idx * PROG_SIZE,complex_output_bf16,mask=output_mask)

def get_rank():
    return int(os.environ["RANK"])
    
def rope_apply(x, grid_sizes, freqs):
    """
    x:          [B, L, N, C].
    grid_sizes: [B, 3].
    freqs:      [M, C // 2].
    """
    s, n, c = x.size(1), x.size(2), x.size(3) // 2
    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :s].to(torch.float64).reshape(
            s, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
                            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        sp_size = 8
        sp_rank = get_rank()
        # freqs_i = pad_freqs(freqs_i, s * sp_size)
        # torch.save(freqs_i,f"rank_{get_rank()}_freqs_i.pt")
        s_per_rank = s
        freqs_i_rank = freqs_i[(sp_rank * s_per_rank):((sp_rank + 1) *
                                                       s_per_rank), :, :]
        x_i = torch.view_as_real(x_i * freqs_i_rank).flatten(2)
        x_i = torch.cat([x_i, x[i, s:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).float()

def all_to_all(x, scatter_dim, gather_dim, group=None, **kwargs):
    """
    `scatter` along one dimension and `gather` along another.
    """
    world_size = 8
    if world_size > 1:
        inputs = [u.contiguous() for u in x.chunk(world_size, dim=scatter_dim)]
        outputs = [torch.empty_like(u) for u in inputs]
        dist.all_to_all(outputs, inputs, group=group, **kwargs)
        x = torch.cat(outputs, dim=gather_dim).contiguous()
    return x

@triton.jit
def triton_all_to_all_4D(data, # bf16
        hs:tl.constexpr,
        hn:tl.constexpr,
        seq_per_rank:tl.constexpr,
        local_rank:tl.constexpr,
        world_size:tl.constexpr,
        iris_buffer,
        heap_bases:tl.tensor):

    input_head_num = hn # 40
    output_head_num = input_head_num // world_size # 5
    input_seq_len = seq_per_rank # 13640
    # output_seq_len = input_seq_len * world_size # 109120
    program_id = tl.program_id(0)
    start_ptr = data + program_id * hn * hs
    for head_idx in tl.range(0,hn,1):
        target_rank = head_idx // output_head_num
        head_idx_in_target_rank = head_idx - (target_rank * output_head_num) # 0-4 per rank
        token_id_in_target_rank = local_rank * input_seq_len + program_id # 0-109120
        offset_in_target_rank = output_head_num * hs * token_id_in_target_rank + \
                        hs * head_idx_in_target_rank
        target_pointers = iris_buffer + offset_in_target_rank + tl.arange(0,hs)
        head_data_offset = head_idx * hs + tl.arange(0,hs)
        value_bf16 = tl.cast(tl.load(start_ptr + head_data_offset,mask = None),tl.bfloat16)
        iris.store(
                pointer = target_pointers,
                value = value_bf16,
                from_rank = local_rank,
                to_rank = target_rank,
                heap_bases = heap_bases,
                mask = None
            )

@triton.jit
def all_to_all_triton2(data, # bf16
        local_rank:tl.constexpr,
        iris_buffer,
        heap_bases:tl.tensor):
    pid = tl.program_id(0)
    local_start_ptr = data + pid * 40 * 128
    for dst_rank in tl.range(0,8):
        local_src_ptrs = local_start_ptr + dst_rank * 128*5 + tl.arange(0,128*8)
        head_data = tl.load(local_src_ptrs,mask=tl.arange(0,128*8) < 128*5)
        dst_1 = iris_buffer + local_rank*13640*128*5 + pid*128*5+tl.arange(0,128*8)
        iris.store(
            pointer = dst_1,
            value = head_data,
            from_rank = local_rank,
            to_rank = dst_rank,
            heap_bases = heap_bases,
            mask=tl.arange(0,128*8) < 128*5
        )

@triton.jit
def all_to_all_triton3(data, # bf16
        PROG_SIZE:tl.constexpr,
        head_num:tl.constexpr,
        seq_per_rank:tl.constexpr,
        local_rank:tl.constexpr,
        world_size:tl.constexpr,
        iris_buffer,
        heap_bases:tl.tensor):

    program_id = tl.program_id(0)
    start_ptr = data + program_id * 40 * 128
    for head_idx in tl.range(0,40,1):
        target_rank = head_idx // 5
        head_idx_in_target_rank = head_idx - (target_rank * 5) # 0-4 per rank
        token_id_in_target_rank = local_rank * 13640 + program_id # 0-109120
        offset_in_target_rank = 5*128*token_id_in_target_rank+128*head_idx_in_target_rank
        target_pointers = iris_buffer + offset_in_target_rank + tl.arange(0,128)
        head_data_offset = head_idx * 128 + tl.arange(0,128)
        complex_output_bf16 = tl.load(start_ptr + head_data_offset,mask = None)
        iris.store(
                pointer = target_pointers, 
                value = complex_output_bf16,
                from_rank = local_rank,
                to_rank = target_rank,
                heap_bases = heap_bases,
                mask = None
            )

@triton.jit
def triton_all_to_all_4D_bf16_backward1(data, # bf16
                    hs:tl.constexpr,
                    in_hn:tl.constexpr,
                    seq_this_rank:tl.constexpr,
                    local_rank:tl.constexpr,
                    world_size:tl.constexpr,
                    iris_buffer,
                    heap_bases:tl.tensor):
    pid = tl.program_id(0) # 当前这个block要把数据写到0-7号rank的这个token id的位置
    output_seq_len = seq_this_rank // world_size
    out_hn = in_hn * world_size
    for target_rank in tl.range(0,world_size):
        rank_stride = output_seq_len * target_rank
        token_id = pid + rank_stride # 当前要写出的数据在本地视角下的token id
        local_rank_data_start_offsets = data + token_id * in_hn * hs + tl.arange(0,hs)
        target_rank_data_start_offsets = iris_buffer + pid * out_hn * hs + (local_rank * in_hn * hs) + tl.arange(0,hs)
        for head_idx in tl.range(0,in_hn):
            local_ptrs = local_rank_data_start_offsets + (head_idx * hs)
            head_data = tl.load(local_ptrs,mask=None)
            remote_ptrs = target_rank_data_start_offsets + (head_idx * hs)
            # write to target rank
            iris.store(
                pointer = remote_ptrs,
                value = head_data,
                from_rank = local_rank,
                to_rank = target_rank,
                heap_bases = heap_bases,
                mask = None
            )


@triton.jit
def triton_all_to_all_4D_bf16_backward_reference(data, # bf16
                    hs:tl.constexpr,
                    in_hn:tl.constexpr,
                    seq_this_rank:tl.constexpr,
                    local_rank:tl.constexpr,
                    world_size:tl.constexpr,
                    iris_buffer,
                    heap_bases:tl.tensor):
    pid = tl.program_id(0) # 当前这个block要把数据写到0-7号rank的这个token id的位置
    output_seq_len = seq_this_rank // world_size
    out_hn = in_hn * world_size
    for target_rank in tl.range(0,world_size):
        rank_stride = output_seq_len * target_rank
        token_id = pid + rank_stride # 当前要写出的数据在本地视角下的token id
        local_rank_data_start_offsets = data + token_id * in_hn * hs + tl.arange(0,hs)
        target_rank_data_start_offsets = iris_buffer + pid * out_hn * hs + (local_rank * in_hn * hs) + tl.arange(0,hs)
        for head_idx in tl.range(0,in_hn):
            local_ptrs = local_rank_data_start_offsets + (head_idx * hs)
            head_data = tl.load(local_ptrs,mask=None)
            remote_ptrs = target_rank_data_start_offsets + (head_idx * hs)
            # write to target rank
            iris.store(
                pointer = remote_ptrs,
                value = head_data,
                from_rank = local_rank,
                to_rank = target_rank,
                heap_bases = heap_bases,
                mask = None
            )

@triton.jit
def triton_all_to_all_4D_bf16_backward11(iris_input_buffer,
                    hs:tl.constexpr,
                    in_hn:tl.constexpr, # 5
                    seq_this_rank:tl.constexpr, # 109120
                    local_rank:tl.constexpr,
                    world_size:tl.constexpr,
                    local_output_bufer,
                    heap_bases:tl.tensor):
    pid = tl.program_id(0) # 当前这个block要把数据写到0-7号rank的这个token id的位置
    output_seq_len = seq_this_rank // world_size
    out_hn = in_hn * world_size
    for target_rank in tl.range(0,world_size):
        remote_data_start_offset = (local_rank * output_seq_len)*(in_hn * hs) + pid * (in_hn * hs)
        local_output_start_offset = pid * out_hn * hs
        for head_idx in tl.range(0,in_hn):
            remote_ptrs = iris_input_buffer + remote_data_start_offset + head_idx * hs + tl.arange(0,hs)
            head_data = iris.load(
                pointer = remote_ptrs,
                to_rank = local_rank,
                from_rank = target_rank,
                heap_bases = heap_bases,
                mask = None
            )
            tl.store(
                pointer=local_output_bufer + local_output_start_offset + target_rank * in_hn * hs + head_idx *hs + tl.arange(0,hs),
                value = head_data,
                mask = None
            )


@triton.jit
def triton_all_to_all_4D_no_sync(iris_input_buffer, # bf16
                    hs:tl.constexpr,
                    in_hn:tl.constexpr,
                    seq_this_rank:tl.constexpr,
                    local_rank:tl.constexpr,
                    world_size:tl.constexpr,
                    local_output_buffer,
                    heap_bases:tl.tensor):
    pid = tl.program_id(0)
    

q = torch.load(f"rank_{rank}_before_rope_q.pt") # fp32
k = torch.load(f"rank_{rank}_before_rope_k.pt")
v = torch.load(f"rank_{rank}_before_rope_v.pt")

freqs_i = torch.load(f"rank_{rank}_freqs_i.pt")
freqs = torch.load(f"rank_{rank}_freqs.pt")
grid_size = torch.load(f"rank_{rank}_grid_sizes.pt")

q_reference = rope_apply(q,grid_size,freqs).to(torch.bfloat16)
k_reference = rope_apply(k,grid_size,freqs).to(torch.bfloat16)
q_after_rope = torch.load(f"rank_{rank}_rope_half_output_before_alltoall_q.pt")
k_after_rope = torch.load(f"rank_{rank}_rope_half_output_before_alltoall_k.pt")
v_after_rope = torch.load(f"rank_{rank}_rope_half_output_before_alltoall_v.pt")

q_alltoall_reference = all_to_all(q_after_rope,scatter_dim=2,gather_dim=1)
# shmem.barrier()
# k_alltoall_reference = all_to_all(k_after_rope,scatter_dim=2,gather_dim=1)
iris_o.copy_(q_alltoall_reference)

q_empty = torch.zeros_like(q_after_rope).to(iris_o.device)


# 测试 all to all backward结果
# triton_all_to_all_4D_bf16_backward11[(13640,1,1)](iris_o,128,5,109120,rank,8,q_empty,shmem.get_heap_bases())
# triton_all_to_all_4D_bf16_backward_reference[(13640,1,1)](q_alltoall_reference,128,5,109120,rank,8,iris_o,shmem.get_heap_bases())
# shmem.barrier()
# print(iris_o - q_after_rope)


triton_all_to_all_4D_bf16_backward11[(13640,1,1)](iris_o,128,5,109120,rank,8,q_empty,shmem.get_heap_bases())
# shmem.barrier()
print((q_empty - q_after_rope).sum())

# if True: # 测试 all to all结果
#     all_to_all_triton[(13640,1,1)](q_after_rope,128,40,13640,rank,8,all_to_all_iris,shmem.get_heap_bases())
#     shmem.barrier()
#     qq=all_to_all_iris.clone()
    
#     print(f"rank {rank} SAME?  {1-torch.abs((qq - q_alltoall_reference)).sum().item()} \n")
    # all_to_all_triton[(13640,1,1)](k_after_rope,128,40,13640,rank,8,all_to_all_iris,shmem.get_heap_bases())
    # kk = all_to_all_iris.clone()
    # print(f"rank {rank} SAME?  {1-torch.abs((kk - k_alltoall_reference)).sum().item()} \n")

# q_alltoall_true = torch.load(f"rank_{rank}_after_alltoall_q.pt")
# k_alltoall_true = torch.load(f"rank_{rank}_after_alltoall_k.pt")
# print(f"rank {rank} SAME?  {1-torch.abs((q_alltoall_true - q_alltoall_reference)).sum().item()} \n")
# print(f"rank {rank} SAME?  {1-torch.abs((k_alltoall_true - k_alltoall_reference)).sum().item()} \n")
# freqs_i = torch.view_as_real(freqs_i)

# rope_triton_kernel_fp16_with_alltoall[(13640,1,1)](q,freqs_i,128,rank,13640,40,all_to_all_iris,8,shmem.get_heap_bases())
# all_to_all_triton[(13640,1,1)](q_after_rope,128,40,13640,rank,8,all_to_all_iris,shmem.get_heap_bases())
# all_to_all_triton[(13640,1,1)](q,128,40,13640,rank,8,all_to_all_iris,shmem.get_heap_bases())
# q=all_to_all(q_after_rope,2,1)
# all_to_all_triton2[(13640,1,1)](q_after_rope,rank,all_to_all_iris,shmem.get_heap_bases())
# print(f"rank {rank} {torch.abs(all_to_all_iris-q_alltoall_true).max().item()}")
# print(torch.abs(all_to_all_iris-q_alltoall_true))
# print(f"rank {rank} SAME?  {1-torch.abs((freqs_i_reference - freqs_i)).sum().item()} \n")

# freqs_i = torch.view_as_real(freqs_i)
# q_triton_output = torch.empty_like(q_after_rope).to(torch.bfloat16)
# k_triton_output = torch.empty_like(k_after_rope).to(torch.bfloat16)
# rope_triton_kernel_fp16[(13640,1,1)](q,freqs_i,q_triton_output,128,rank,13640,40)
# q=all_to_all(q_triton_output,2,1)
# print(f"rank {rank} SAME?  {1-torch.abs((q - q_alltoall_reference)).sum().item()} \n")
# # rope_triton_kernel_fp16[(13640,1,1)](k,freqs_i,k_triton_output,128,rank,13640,40)
# print(f"rank {rank} SAME?  {1-torch.abs((q_triton_output - q_after_rope)).sum().item()} \n")
# yy = torch.abs(q_triton_output - q_after_rope).reshape(13640*40,128)
# count = 0
# for i in yy:
#     if i.max() > 0.0000001:
#         print(f"{count = },i={i},{i.max() = }")
#     count +=1



