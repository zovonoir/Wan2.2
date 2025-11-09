# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
import torch.cuda.amp as amp

from ..modules.model import sinusoidal_embedding_1d
from .ulysses import distributed_attention
from .util import gather_forward, get_rank, get_world_size
import triton
import triton.language as tl
import iris
from ..modules.attention import flash_attention
from .util import all_to_all

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
    output_ptr_block = output_ptr + program_id * PROG_SIZE * head_num + tl.arange(0,PROG_SIZE)
    output_mask = tl.arange(0,PROG_SIZE) < PROG_SIZE
    # # alltoall 4D输出形状
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
        tl.store(output_ptr_block + head_idx * PROG_SIZE,complex_output_bf16,mask=output_mask)
        # 这个head数据需要写到哪张卡哪个地址上?

        # target_rank = head_idx // output_head_num
        # head_idx_in_target_rank = head_idx - ((head_idx // output_head_num) * output_head_num) # 0-4 per rank
        # token_id_in_target_rank = sp_rank * input_seq_len + program_id # 0-109120
        # offset_in_target_rank = output_head_num * PROG_SIZE * token_id_in_target_rank + PROG_SIZE * head_idx_in_target_rank
        # target_pointers = iris_buffer + offset_in_target_rank + tl.arange(0,PROG_SIZE)
        # iris.store(
        #         pointer = target_pointers, # iris_buffer + head_idx*PROG_SIZE +tl.arange(0,PROG_SIZE), #target_pointers,
        #         value = complex_output_bf16,
        #         from_rank = sp_rank,
        #         to_rank = target_rank,
        #         heap_bases = heap_bases,
        #         mask = None
            # )



@triton.jit
def rope_triton_kernel_bf16_alltoall_4D(qk_ptr, freqs_ptr,
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
    freqs_rank_offset = hs * local_rank * seq_len_per_rank # freq当前rank在freq中的偏移量
    freq_rank_start_ptr = freqs_ptr + freqs_rank_offset # 这个rank的起始指针
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
    vx1_ptr_block_even = qk_ptr + program_id * hs * in_hn + even_offset # 当前BLOCK处理数据的第0个head的起始地址 + 每个数据偏移量
    vx1_ptr_block_odd = qk_ptr + program_id * hs * in_hn + odd_offset
    output_mask = tl.arange(0,hs) < hs
    # alltoall 4D输出形状
    input_head_num = in_hn
    output_head_num = input_head_num // world_size
    input_seq_len = seq_len_per_rank
    output_seq_len = input_seq_len * world_size
    # [1,input_seq_len,input_head_num,head_size] -> [1,input_seq_len * world_size,input_head_num / world_size,head_size]

    for head_idx in tl.range(0,in_hn,1): # 在head 维度循环,每个program计算一个token的数据量
        # load single head for x
        vx1_fp64 = tl.cast(tl.load(vx1_ptr_block_even + head_idx * hs,mask=even_mask,other=0.0),tl.float64) # mask可能不严谨
        vx2_fp64 = tl.cast(tl.load(vx1_ptr_block_odd + head_idx * hs,mask=odd_mask,other=0.0),tl.float64)
        # complex_output = tl.cast(vx1*vy1 + coe*vx2*vy2,tl.float32) # 一个head的输出
        complex_output_bf16 = tl.cast(vx1_fp64*vy1_fp64 + coe*vx2_fp64*vy2_fp64,tl.bfloat16) # rope输出是float32格式,但是后续做alltoall之前又被转换成了bfloat16,因此这里提前转换再写入
        # tl.store(output_ptr_block + head_idx * PROG_SIZE,complex_output,mask=output_mask)
        # which rank is the target rank of this head?

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
def triton_all_to_all_4D_bf16_forward(data, # bf16
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
def triton_all_to_all_4D_bf16_backward(data, # bf16
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
            
    

def pad_freqs(original_tensor, target_len):
    seq_len, s1, s2 = original_tensor.shape
    pad_size = target_len - seq_len
    padding_tensor = torch.ones(
        pad_size,
        s1,
        s2,
        dtype=original_tensor.dtype,
        device=original_tensor.device)
    padded_tensor = torch.cat([original_tensor, padding_tensor], dim=0)
    return padded_tensor


@torch.amp.autocast('cuda', enabled=False)
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
        sp_size = get_world_size()
        sp_rank = get_rank()
        freqs_i = pad_freqs(freqs_i, s * sp_size)
        # torch.save(freqs_i,f"rank_{get_rank()}_freqs_i.pt")
        s_per_rank = s
        freqs_i_rank = freqs_i[(sp_rank * s_per_rank):((sp_rank + 1) *
                                                       s_per_rank), :, :]
        x_i = torch.view_as_real(x_i * freqs_i_rank).flatten(2)
        x_i = torch.cat([x_i, x[i, s:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).float()


def sp_dit_forward(
    self,
    x,
    t,
    context,
    seq_len,
    y=None,
):
    """
    x:              A list of videos each with shape [C, T, H, W].
    t:              [B].
    context:        A list of text embeddings each with shape [L, C].
    """
    if self.model_type == 'i2v':
        assert y is not None
    # params
    device = self.patch_embedding.weight.device
    if self.freqs.device != device:
        self.freqs = self.freqs.to(device)

    if y is not None:
        x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

    # embeddings
    x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
    grid_sizes = torch.stack(
        [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
    x = [u.flatten(2).transpose(1, 2) for u in x]
    seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
    assert seq_lens.max() <= seq_len
    x = torch.cat([
        torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1)
        for u in x
    ])

    # time embeddings
    if t.dim() == 1:
        t = t.expand(t.size(0), seq_len)
    with torch.amp.autocast('cuda', dtype=torch.float32):
        bt = t.size(0)
        t = t.flatten()
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim,
                                    t).unflatten(0, (bt, seq_len)).float())
        e0 = self.time_projection(e).unflatten(2, (6, self.dim))
        assert e.dtype == torch.float32 and e0.dtype == torch.float32

    # context
    context_lens = None
    context = self.text_embedding(
        torch.stack([
            torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
            for u in context
        ]))

    # Context Parallel
    x = torch.chunk(x, get_world_size(), dim=1)[get_rank()]
    e = torch.chunk(e, get_world_size(), dim=1)[get_rank()]
    e0 = torch.chunk(e0, get_world_size(), dim=1)[get_rank()]

    # arguments
    kwargs = dict(
        e=e0,
        seq_lens=seq_lens,
        grid_sizes=grid_sizes, # not useful
        freqs=self.freqs if not hasattr(self,"freqs_i") else (self.freqs_i,self.shmem_handle,self.iris_buffer_list),
        context=context,
        context_lens=context_lens)

    for block in self.blocks:
        x = block(x, **kwargs)

    # head
    x = self.head(x, e)

    # Context Parallel
    x = gather_forward(x, dim=1)

    # unpatchify
    x = self.unpatchify(x, grid_sizes)
    return [u.float() for u in x]

def sp_attn_forward(self, x, seq_lens, grid_sizes, freqs_i, dtype=torch.bfloat16):
    assert isinstance(freqs_i,tuple)
    global stream_q,stream_k,stream_v
    freqs_i,shmem_handle,iris_buffer_list = freqs_i
    iris_q,iris_k,iris_v,iris_o = iris_buffer_list

    b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
    half_dtypes = (torch.float16, torch.bfloat16)

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # query, key, value function
    def qkv_fn(x):
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)
        return q, k, v

    q, k, v = qkv_fn(x)
    
    if False:
        q = rope_apply(q, grid_sizes, freqs)
        k = rope_apply(k, grid_sizes, freqs)

        x = distributed_attention(
            half(q),
            half(k),
            half(v),
            seq_lens,
            window_size=self.window_size,
        )

        q = all_to_all(q, scatter_dim=2, gather_dim=1)
        k = all_to_all(k, scatter_dim=2, gather_dim=1)
        v = all_to_all(v, scatter_dim=2, gather_dim=1)


    if False: # 这一段被证明是完全有效的
        q_buffer = half(torch.empty_like(q))
        k_buffer = half(torch.empty_like(k))
        rope_triton_kernel_fp16[(sp_seq_len,1,1)](q,freqs_i,q_buffer,hs,rank,sp_seq_len,hn)
        rope_triton_kernel_fp16[(sp_seq_len,1,1)](k,freqs_i,k_buffer,hs,rank,sp_seq_len,hn)
        q=q_buffer
        k=k_buffer
        v=half(v)
        q = all_to_all(q, scatter_dim=2, gather_dim=1)
        k = all_to_all(k, scatter_dim=2, gather_dim=1)
        v = all_to_all(v, scatter_dim=2, gather_dim=1)

    if True: # 但是这一段带上alltoall的功能就失效
        bs = q.shape[0]
        hs = q.shape[-1]
        rank = get_rank()
        sp_seq_len = q.shape[1]
        hn = q.shape[2]
        world_size = get_world_size()
        heap_bases = shmem_handle.get_heap_bases()
        # q_alltoall_buffer = torch.empty([bs,sp_seq_len * world_size,hn//world_size,hs],dtype = dtype,device=x.device)
        # shmem_handle.barrier()
        # q = iris_q.clone() # iris_buffer_tensor.clone().reshape([bs,world_size*sp_seq_len,hn // world_size,hs])

        
        # shmem_handle.barrier()
        # k = iris_k.clone() #iris_buffer_tensor.clone().reshape([bs,world_size*sp_seq_len,hn // world_size,hs])
        # v=half(v)
        # v=all_to_all(v,2,1)

        rope_triton_kernel_bf16_alltoall_4D[(sp_seq_len,1,1)](q,freqs_i,hs,rank,sp_seq_len,hn, iris_q,world_size,heap_bases)
        rope_triton_kernel_bf16_alltoall_4D[(sp_seq_len,1,1)](k,freqs_i,hs,rank,sp_seq_len,hn, iris_k,world_size,heap_bases)
        triton_all_to_all_4D_bf16_forward[(sp_seq_len,1,1)](v,hs,hn,sp_seq_len,rank,world_size,iris_v,heap_bases)
        shmem_handle.barrier()

        q = iris_q
        k = iris_k
        v = iris_v


    if False: # q执行alltoall kernel,k执行rope kernel
        bs = q.shape[0]
        hs = q.shape[-1]
        rank = get_rank()
        sp_seq_len = q.shape[1]
        hn = q.shape[2]
        world_size = get_world_size()
        heap_bases = shmem_handle.get_heap_bases()
        q_alltoall_buffer = torch.empty([bs,sp_seq_len * world_size,hn//world_size,hs],dtype = dtype,device=x.device)
        rope_triton_kernel_bf16_alltoall_4D[(sp_seq_len,1,1)](q,freqs_i,hs,rank,sp_seq_len,hn, iris_buffer_tensor,world_size,heap_bases)
        q = q_alltoall_buffer.copy_(iris_buffer_tensor) # iris_buffer_tensor.clone().reshape([bs,world_size*sp_seq_len,hn // world_size,hs])

        k_buffer = half(torch.empty_like(k))
        rope_triton_kernel_fp16[(sp_seq_len,1,1)](k,freqs_i,k_buffer,hs,rank,sp_seq_len,hn)
        k=k_buffer
        k = all_to_all(k, scatter_dim=2, gather_dim=1)

        v=half(v)
        v = all_to_all(v, scatter_dim=2, gather_dim=1)

    x = flash_attention(
        q,
        k,
        v,
        k_lens=seq_lens,
        window_size=self.window_size,
    )

    # x = all_to_all(x, scatter_dim=1, gather_dim=2)
    triton_all_to_all_4D_bf16_backward[(sp_seq_len,1,1)](x,hs,hn//world_size,sp_seq_len*world_size,rank,world_size,iris_o,heap_bases)
    iris_o = iris_o.flatten(2)
    shmem_handle.barrier()
   
    x = self.o(iris_o)
    return x

def sp_attn_forward1(self, x, seq_lens, grid_sizes, freqs, dtype=torch.bfloat16):
    b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
    half_dtypes = (torch.float16, torch.bfloat16)

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # query, key, value function
    def qkv_fn(x):
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)
        return q, k, v

    q, k, v = qkv_fn(x)
    
    # 保存qkv
    rank = get_rank()
    # torch.save(q,f"rank_{rank}_before_rope_q.pt")
    # torch.save(k,f"rank_{rank}_before_rope_k.pt")
    # torch.save(v,f"rank_{rank}_before_rope_v.pt")
    # torch.save(grid_sizes,f"rank_{rank}_grid_sizes.pt")
    # torch.save(freqs,f"rank_{rank}_freqs.pt")

    q = rope_apply(q, grid_sizes, freqs)
    k = rope_apply(k, grid_sizes, freqs)
    q=half(q)
    k=half(k)
    v=half(v)
    # 此处将qkv tensor保存下来,然后assert
    # torch.save(q,f"rank_{rank}_rope_half_output_before_alltoall_q.pt")
    # torch.save(k,f"rank_{rank}_rope_half_output_before_alltoall_k.pt")
    # torch.save(v,f"rank_{rank}_rope_half_output_before_alltoall_v.pt")
    # assert 0,"saving complete!"

    x = distributed_attention(
        q,k,v,
        seq_lens,
        window_size=self.window_size,
        rank = get_rank()
    )

    # output
    x = x.flatten(2)
    x = self.o(x)
    return x
