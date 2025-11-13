import triton
import triton.language as tl
import iris

@triton.jit
def rope_triton_kernel_fp16(qk_ptr, freqs_ptr, 
                    output_ptr,
                    PROG_SIZE:tl.constexpr,# = head_size
                    sp_rank:tl.constexpr, # [0-7]
                    s_per_rank:tl.constexpr, # 13640
                    head_num:tl.constexpr, # 40
                    # alltoall
                    iris_buffer,
                    world_size:tl.constexpr,
                    heap_bases:tl.tensor
                    ):
    program_id = tl.program_id(0)
    tl.static_assert(PROG_SIZE > 0 and (PROG_SIZE & (PROG_SIZE - 1)) == 0,f"PROG_SIZE only support power of 2!current is {PROG_SIZE}")
    freqs_rank_offset = PROG_SIZE * sp_rank * s_per_rank
    freq_rank_start_ptr = freqs_ptr + freqs_rank_offset
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
    vx1_ptr_block_even = qk_ptr + program_id * PROG_SIZE * head_num + even_offset
    vx1_ptr_block_odd = qk_ptr + program_id * PROG_SIZE * head_num + odd_offset
    output_ptr_block = output_ptr + program_id * PROG_SIZE * head_num + tl.arange(0,PROG_SIZE)
    output_mask = tl.arange(0,PROG_SIZE) < PROG_SIZE
    # # alltoall 4D输出形状
    input_head_num = head_num
    output_head_num = input_head_num // world_size
    input_seq_len = s_per_rank
    output_seq_len = input_seq_len * world_size
    # [1,input_seq_len,input_head_num,head_size] -> [1,input_seq_len * world_size,input_head_num / world_size,head_size]

    for head_idx in tl.range(0,head_num,1):
        # load single head for x
        vx1_fp64 = tl.cast(tl.load(vx1_ptr_block_even + head_idx * PROG_SIZE,mask=even_mask,other=0.0),tl.float64) # mask可能不严谨
        vx2_fp64 = tl.cast(tl.load(vx1_ptr_block_odd + head_idx * PROG_SIZE,mask=odd_mask,other=0.0),tl.float64)
        # complex_output = tl.cast(vx1*vy1 + coe*vx2*vy2,tl.float32) # 一个head的输出
        complex_output_bf16 = tl.cast(vx1_fp64*vy1_fp64 + coe*vx2_fp64*vy2_fp64,tl.bfloat16) # NOTE: convert to bf16
        tl.store(output_ptr_block + head_idx * PROG_SIZE,complex_output_bf16,mask=output_mask)

@triton.jit
def rope_alltoall_4D_bf16_forward(qk_ptr, freqs_ptr,
                    hs:tl.constexpr,# head size
                    local_rank:tl.constexpr, # [0-7]
                    seq_len_per_rank:tl.constexpr, # 13640
                    in_hn:tl.constexpr, # 输入的head num 40
                    # alltoall
                    iris_buffer,
                    world_size:tl.constexpr,
                    heap_bases:tl.tensor
                    ):
    # 每个program负责一个token,共40*128=5120个float32相乘
    program_id = tl.program_id(0)
    tl.static_assert(hs > 0 and (hs & (hs - 1)) == 0,f"PROG_SIZE only support power of 2!current is {hs}")
    freqs_rank_offset = hs * local_rank * seq_len_per_rank
    freq_rank_start_ptr = freqs_ptr + freqs_rank_offset
    freq_program_start_ptr = freq_rank_start_ptr + program_id * hs
    even_offset = 2 * (tl.arange(0,hs) // 2) # program_start + [0,0,2,2,4,4,...126,126]
    odd_offset = even_offset + 1 # program_start + [1,1,3,3,5,5,...,127,127]
    freqs_offset = tl.arange(0,hs) #  [0,1,2,3,4,5...],freq只能load128个数据,并且复用
    freqs_offset_swap = tl.arange(0,hs) ^ 0x0001 # [1,0,3,2,5,4...]
    coe = 2*(tl.arange(0,hs) % 2) - 1 # [-1,1,-1,1,-1,1...]
    even_mask = even_offset < hs # all true
    odd_mask = odd_offset < hs
    freqs_mask = freqs_offset < hs
    freqs_mask_swap = freqs_offset_swap < hs
    vy1_fp64 = tl.cast(tl.load(freq_program_start_ptr + freqs_offset,mask=freqs_mask,other=0.0),tl.float64)
    vy2_fp64 = tl.cast(tl.load(freq_program_start_ptr + freqs_offset_swap,mask=freqs_mask_swap,other=0.0),tl.float64)
    vx1_ptr_block_even = qk_ptr + program_id * hs * in_hn + even_offset
    vx1_ptr_block_odd = qk_ptr + program_id * hs * in_hn + odd_offset
    output_mask = tl.arange(0,hs) < hs
    # alltoall 4D输出形状
    input_head_num = in_hn
    output_head_num = input_head_num // world_size
    input_seq_len = seq_len_per_rank
    output_seq_len = input_seq_len * world_size
    # [1,input_seq_len,input_head_num,head_size] -> [1,input_seq_len * world_size,input_head_num / world_size,head_size]

    for head_idx in tl.range(0,in_hn,1): # for loop at head dimensionm,each program is responsible for single token
        # load single head for x
        vx1_fp64 = tl.cast(tl.load(vx1_ptr_block_even + head_idx * hs,mask=even_mask,other=0.0),tl.float64) # is mask correct?
        vx2_fp64 = tl.cast(tl.load(vx1_ptr_block_odd + head_idx * hs,mask=odd_mask,other=0.0),tl.float64)
        complex_output_bf16 = tl.cast(vx1_fp64*vy1_fp64 + coe*vx2_fp64*vy2_fp64,tl.bfloat16) # NOTE: convert to bfloat16 before all to all procedure

        target_rank = head_idx // output_head_num
        head_idx_in_target_rank = head_idx - ((head_idx // output_head_num) * output_head_num) # 0-4 per rank
        token_id_in_target_rank = local_rank * input_seq_len + program_id # 0-109120
        offset_in_target_rank = output_head_num * hs * token_id_in_target_rank + hs * head_idx_in_target_rank
        target_pointers = iris_buffer + offset_in_target_rank + tl.arange(0,hs)
        iris.store(
                pointer = target_pointers,
                value = complex_output_bf16,
                from_rank = local_rank,
                to_rank = target_rank,
                heap_bases = heap_bases,
                mask = None
            )

@triton.jit
def all_to_all_4D_bf16_forward(data, # bf16
        hs:tl.constexpr,
        in_hn:tl.constexpr,
        seq_len_per_rank:tl.constexpr,
        local_rank:tl.constexpr,
        world_size:tl.constexpr,
        iris_buffer,
        heap_bases:tl.tensor):

    input_head_num = in_hn # 40
    output_head_num = input_head_num // world_size # 5
    input_seq_len = seq_len_per_rank # 13640
    program_id = tl.program_id(0)
    start_ptr = data + program_id * in_hn * hs
    for head_idx in tl.range(0,in_hn,1):
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
def triton_all_to_all_4D_bf16_backward_deprecated(data, # bf16
                    hs:tl.constexpr,
                    in_hn:tl.constexpr,
                    seq_this_rank:tl.constexpr,
                    local_rank:tl.constexpr,
                    world_size:tl.constexpr,
                    iris_buffer,
                    heap_bases:tl.tensor):
    pid = tl.program_id(0) # current program need to write data to rank 0-7 at the tokenid located
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
def all_to_all_4D_bf16_backward(iris_input_buffer,
                    hs:tl.constexpr,
                    in_hn:tl.constexpr, # 5
                    seq_this_rank:tl.constexpr, # 109120
                    local_rank:tl.constexpr,
                    world_size:tl.constexpr,
                    local_output_bufer,
                    heap_bases:tl.tensor):
    pid = tl.program_id(0)
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