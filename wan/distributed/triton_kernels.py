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
    # even_mask = even_offset < hs # all true
    # odd_mask = odd_offset < hs
    # freqs_mask = freqs_offset < hs
    # freqs_mask_swap = freqs_offset_swap < hs
    # vy1_fp64 = tl.cast(tl.load(freq_program_start_ptr + freqs_offset,mask=freqs_mask,other=0.0),tl.float64)
    # vy2_fp64 = tl.cast(tl.load(freq_program_start_ptr + freqs_offset_swap,mask=freqs_mask_swap,other=0.0),tl.float64)
    vy1_fp64 = tl.cast(tl.load(freq_program_start_ptr + freqs_offset,mask=None),tl.float64)
    vy2_fp64 = tl.cast(tl.load(freq_program_start_ptr + freqs_offset_swap,mask=None),tl.float64)
    vx1_ptr_block_even = qk_ptr + program_id * hs * in_hn + even_offset
    vx1_ptr_block_odd = qk_ptr + program_id * hs * in_hn + odd_offset
    output_mask = tl.arange(0,hs) < hs
    # alltoall 4D输出形状
    input_head_num = in_hn
    output_head_num = input_head_num // world_size
    input_seq_len = seq_len_per_rank
    output_seq_len = input_seq_len * world_size
    # [1,input_seq_len,input_head_num,head_size] -> [1,input_seq_len * world_size,input_head_num / world_size,head_size]

    for head_idx in tl.range(0,in_hn,1,num_stages=4): # for loop at head dimensionm,each program is responsible for single token
        # load single head for x
        # vx1_fp64 = tl.cast(tl.load(vx1_ptr_block_even + head_idx * hs,mask=even_mask,other=0.0),tl.float64) # is mask correct?
        # vx2_fp64 = tl.cast(tl.load(vx1_ptr_block_odd + head_idx * hs,mask=odd_mask,other=0.0),tl.float64)
        vx1_fp64 = tl.cast(tl.load(vx1_ptr_block_even + head_idx * hs,mask=None),tl.float64) # is mask correct?
        vx2_fp64 = tl.cast(tl.load(vx1_ptr_block_odd + head_idx * hs,mask=None),tl.float64)
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
    for target_rank in tl.range(0,world_size,num_stages=4):
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
def __alltoall_load_single_token_data_from_target_rank(
                    token_id,
                    iris_input_buffer,
                    hs:tl.constexpr,
                    in_hn:tl.constexpr, # 5
                    seq_this_rank:tl.constexpr, # 109120
                    local_rank:tl.constexpr,
                    world_size:tl.constexpr,
                    local_output_bufer,
                    target_rank:tl.constexpr,
                    heap_bases:tl.tensor):
    pid = token_id #tl.program_id(0) # 这就是token id
    output_seq_len = seq_this_rank // world_size
    out_hn = in_hn * world_size
    remote_data_start_offset = (local_rank * output_seq_len)*(in_hn * hs) + pid * (in_hn * hs)
    local_output_start_offset = pid * out_hn * hs
    for head_idx in tl.static_range(0,in_hn):
        remote_ptrs = iris_input_buffer + remote_data_start_offset + head_idx * hs + tl.arange(0,hs)
        # # 测试一下从本地load写到远程，非常快
        head_data = tl.load(local_output_bufer + local_output_start_offset + target_rank * in_hn * hs + head_idx *hs + tl.arange(0,hs),
                    mask=None)
        iris.store(
                pointer = remote_ptrs,
                value = head_data,
                from_rank = local_rank,
                to_rank = target_rank,
                heap_bases = heap_bases,
                mask = None
            )
        ####################################################################################################################################
        # head_data = iris.load(
        #     pointer = remote_ptrs,
        #     to_rank = local_rank,
        #     from_rank = target_rank,
        #     heap_bases = heap_bases,
        #     mask = None
        # )
        # tl.store(
        #     pointer=local_output_bufer + local_output_start_offset + target_rank * in_hn * hs + head_idx *hs + tl.arange(0,hs),
        #     value = head_data,
        #     mask = None
        # )
        ####################################################################################################################################
    # remote_ptrs1 = iris_input_buffer + remote_data_start_offset + 0 * 128 + tl.arange(0,128)
    # remote_ptrs2 = iris_input_buffer + remote_data_start_offset + 1 * 128 + tl.arange(0,128)
    # remote_ptrs3 = iris_input_buffer + remote_data_start_offset + 2 * 128 + tl.arange(0,128)
    # remote_ptrs4 = iris_input_buffer + remote_data_start_offset + 3 * 128 + tl.arange(0,128)
    # remote_ptrs5 = iris_input_buffer + remote_data_start_offset + 4 * 128 + tl.arange(0,128)
    # head_data1 = iris.load(pointer = remote_ptrs1,to_rank = local_rank,from_rank = target_rank,heap_bases = heap_bases,mask = None)
    # head_data2 = iris.load(pointer = remote_ptrs2,to_rank = local_rank,from_rank = target_rank,heap_bases = heap_bases,mask = None)
    # head_data3 = iris.load(pointer = remote_ptrs3,to_rank = local_rank,from_rank = target_rank,heap_bases = heap_bases,mask = None)
    # head_data4 = iris.load(pointer = remote_ptrs4,to_rank = local_rank,from_rank = target_rank,heap_bases = heap_bases,mask = None)
    # head_data5 = iris.load(pointer = remote_ptrs5,to_rank = local_rank,from_rank = target_rank,heap_bases = heap_bases,mask = None)
    # tl.store(pointer=local_output_bufer + local_output_start_offset + target_rank * 5 * 128 + 0 * 128 + tl.arange(0,128),value = head_data1,mask = None)
    # tl.store(pointer=local_output_bufer + local_output_start_offset + target_rank * 5 * 128 + 1 * 128 + tl.arange(0,128),value = head_data2,mask = None)
    # tl.store(pointer=local_output_bufer + local_output_start_offset + target_rank * 5 * 128 + 2 * 128 + tl.arange(0,128),value = head_data3,mask = None)
    # tl.store(pointer=local_output_bufer + local_output_start_offset + target_rank * 5 * 128 + 3 * 128 + tl.arange(0,128),value = head_data4,mask = None)
    # tl.store(pointer=local_output_bufer + local_output_start_offset + target_rank * 5 * 128 + 4 * 128 + tl.arange(0,128),value = head_data5,mask = None)

@triton.jit
def __alltoall_store_single_token_data_to_target_rank(
                                    token_id, # 1-109120
                                    iris_input_buffer,
                                    hs:tl.constexpr,
                                    in_hn:tl.constexpr, # 5
                                    seq_this_rank:tl.constexpr, # 109120
                                    local_rank:tl.constexpr,
                                    world_size:tl.constexpr,
                                    remote_output_buffer, # [1,13640,40,128]
                                    heap_bases:tl.tensor):
    target_rank = token_id // ( 13640 )
    local_data_offset = token_id * 5 * 128
    remote_data_offset = token_id % (13640)
    for head_idx in tl.range(0,5):
        local_ptrs = iris_input_buffer + local_data_offset + head_idx * 128 + tl.arange(0,128)
        remote_ptrs = remote_output_buffer + remote_data_offset * 40 * 128 + local_rank * 5 * 128 + head_idx * 128 + tl.arange(0,128)
        value = tl.load(local_ptrs,mask=None)
        iris.store(
                pointer = remote_ptrs,
                value = value,
                from_rank = local_rank,
                to_rank = target_rank,
                heap_bases = heap_bases,
                mask = None
            )


# @triton.jit
# def __alltoall_load_single_token_data_from_all_rank(
#                     token_id,
#                     iris_input_buffer,
#                     hs:tl.constexpr,
#                     in_hn:tl.constexpr, # 5
#                     seq_this_rank:tl.constexpr, # 109120
#                     local_rank:tl.constexpr,
#                     world_size:tl.constexpr,
#                     local_output_bufer,
#                     lock_base,lock_offset,
#                     heap_bases:tl.tensor):

#     # token_id = tl.program_id(0)
#     # iris.atomic_cas(pointer = lock_base + lock_offset, 
#     #                 cmp = 0, val = 1, 
#     #                 from_rank = local_rank, 
#     #                 to_rank = local_rank, 
#     #                 heap_bases=heap_bases,
#     #                 sem = "release")

#     finished_flags = 0
#     mask = (1 << world_size) - 1 # 1111 1111 
#     all_finished = ((finished_flags & mask) == mask)

#     # load local data first
#     __alltoall_load_single_token_data_from_target_rank(
#         token_id,
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
#                     __alltoall_load_single_token_data_from_target_rank(
#                         token_id,
#                         iris_input_buffer,hs,in_hn,seq_this_rank,
#                         local_rank,world_size,local_output_bufer,
#                         target_rank,heap_bases)
#                     finished_flags = finished_flags | (1 << target_rank)
#         all_finished = ((finished_flags & mask) == mask)

@triton.jit
def notify(lock_base,lock_offset,
                    heap_bases:tl.tensor,local_rank:tl.constexpr):
    for rank in tl.static_range(0,8):
            iris.store(
            pointer = lock_base + lock_offset*8 + local_rank,
            value = 1,
            from_rank = local_rank,
            to_rank = rank,
            heap_bases = heap_bases,
            mask = None
        )


# def get_all_configs():
#     all_configs = []
#     for num_stages in [1,2,3,4,5]:
#         for num_warps in [1,2,4,8,16]:
#             all_configs.append(triton.Config({"num_stages":num_stages},num_warps=num_warps))
#     return all_configs

# @triton.autotune(
#     configs = get_all_configs(),
#     key=['seq_this_rank']
# )
@triton.jit
def alltoallbackward(
                    iris_input_buffer,
                    hs:tl.constexpr,
                    in_hn:tl.constexpr, # 5
                    seq_this_rank:tl.constexpr, # 109120
                    local_rank:tl.constexpr,
                    world_size:tl.constexpr,
                    local_output_bufer,
                    lock_base,lock_offset,
                    heap_bases:tl.tensor):
    # pid = tl.program_id(0)
    # if pid == 0:
    #     # tl.store(lock_base + lock_offset,1,mask=None)
    #     for rank in tl.static_range(0,8):
    #         iris.store(
    #         pointer = lock_base + lock_offset*8 + local_rank,
    #         value = 1,
    #         from_rank = local_rank,
    #         to_rank = rank,
    #         heap_bases = heap_bases,
    #         mask = None
    #     )
    

    output_seq_len = seq_this_rank // world_size
    start_token_id = tl.program_id(0)
    token_stride = tl.num_programs(0)
    # 先load本地数据
    for token_id in tl.range(start_token_id,output_seq_len,token_stride):
        __alltoall_load_single_token_data_from_target_rank(
            token_id,
            iris_input_buffer,hs,in_hn,seq_this_rank,
            local_rank,world_size,local_output_bufer,
            local_rank,heap_bases)
    
    # 尝试加载其他卡的数据,哪张卡好了就加载哪张卡
    finished_flags:tl.constexpr = 0
    finished_flags = finished_flags | (1 << local_rank) # 将本地数据对应bit设置为1
    mask = (1 << world_size) - 1 # 1111 1111 
    all_finished = ((finished_flags & mask) == mask)
    

    while not all_finished:
        for target_rank in tl.range(0,world_size,num_stages=2):
            # 检查这个rank是不是已经结束了
            rank_finished = ((finished_flags >> target_rank) & 1) == 1
            if not rank_finished:
                # 没有结束就继续检查这个rank上的任务有没有开始
                # lock_released = (iris.atomic_cas(
                #     pointer = lock_base + lock_offset, cmp = 1, val = 1, 
                #     from_rank = local_rank, to_rank = target_rank, 
                #     heap_bases=heap_bases,sem="acquire") == 1)
                # lock_released = (tl.load(lock_base + lock_offset * 8 + target_rank) == 1)
                lock_released = tl.atomic_xor(lock_base + lock_offset * 8 + target_rank,0x0001) == 0
                # value = iris.load(pointer = lock_base + lock_offset,to_rank = local_rank,from_rank = target_rank,heap_bases = heap_bases,mask = None)
                if lock_released:
                    # 已经开始那就可以安全的加载数据
                    # 从这张卡一次性加载全部数据
                    for token_id in tl.range(start_token_id,output_seq_len,token_stride):
                        __alltoall_load_single_token_data_from_target_rank(
                            token_id,
                            iris_input_buffer,hs,in_hn,seq_this_rank,
                            local_rank,world_size,local_output_bufer,
                            target_rank,heap_bases)
                    # 把这张卡标记为完成
                    finished_flags = finished_flags | (1 << target_rank)
                    all_finished = ((finished_flags & mask) == mask)
