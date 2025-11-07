import torch
import torch.distributed as dist
import triton
import triton.language as tl
import iris
import os


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


rank = int(os.environ["RANK"])
world_size =  int(os.environ["WORLD_SIZE"])
dist.init_process_group(
    backend="nccl",
    device_id=torch.device(f"cuda:{rank}"),
    world_size=world_size)
shmem = iris.iris(1024*1024*1024*4)
all_to_all_iris = shmem.zeros([1,13640*8,40//8,128],dtype=torch.bfloat16,device="cuda")

q = torch.load(f"rank_{rank}_before_rope_q.pt")
k = torch.load(f"rank_{rank}_before_rope_k.pt")
v = torch.load(f"rank_{rank}_before_rope_v.pt")
print(q.dtype,k.dtype,v.dtype)
freqs_i = torch.load(f"rank_{rank}_freqs_i.pt")
grid_size = torch.load(f"rank_{rank}_grid_sizes.pt")

q_after_rope = torch.load(f"rank_{rank}_rope_half_output_q.pt")
k_after_rope = torch.load(f"rank_{rank}_rope_half_output_k.pt")
v_after_rope = torch.load(f"rank_{rank}_rope_half_output_v.pt")
freqs_i = torch.view_as_real(freqs_i)
q_triton_output = torch.zeros_like(q).to(torch.bfloat16)
k_triton_output = torch.zeros_like(k).to(torch.bfloat16)
rope_triton_kernel_fp16[(13640,1,1)](q,freqs_i,q_triton_output,128,rank,13640,40)
rope_triton_kernel_fp16[(13640,1,1)](k,freqs_i,k_triton_output,128,rank,13640,40)
print(f"rank {rank} SAME?  {1-torch.abs((q_triton_output - q_after_rope)).mean().item()} \n")