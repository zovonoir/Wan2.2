# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
import torch.cuda.amp as amp

from ..modules.model import sinusoidal_embedding_1d
from .ulysses import distributed_attention
from .util import gather_forward, get_rank, get_world_size
from .triton_kernels import *
from ..modules.attention import flash_attention
from .util import all_to_all
import torch.distributed as dist

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
    iris_q,iris_k,iris_v,iris_o,attn_buffer = iris_buffer_list

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

    bs = q.shape[0]
    hs = q.shape[-1]
    rank = get_rank()
    sp_seq_len = q.shape[1]
    hn = q.shape[2]
    world_size = get_world_size()
    heap_bases = shmem_handle.get_heap_bases()
    rope_triton_kernel_bf16_alltoall_4D[(sp_seq_len,1,1)](q,freqs_i,hs,rank,sp_seq_len,hn, iris_q,world_size,heap_bases)
    rope_triton_kernel_bf16_alltoall_4D[(sp_seq_len,1,1)](k,freqs_i,hs,rank,sp_seq_len,hn, iris_k,world_size,heap_bases)
    triton_all_to_all_4D_bf16_forward[(sp_seq_len,1,1)](v,hs,hn,sp_seq_len,rank,world_size,iris_v,heap_bases)
    shmem_handle.barrier()

    # q = torch.empty([bs,sp_seq_len*world_size,hn//world_size,hs],dtype=iris_q.dtype,device=iris_q.device).copy_(iris_q)
    # k = torch.empty([bs,sp_seq_len*world_size,hn//world_size,hs],dtype=iris_q.dtype,device=iris_q.device).copy_(iris_k)
    # v = torch.empty([bs,sp_seq_len*world_size,hn//world_size,hs],dtype=iris_q.dtype,device=iris_q.device).copy_(iris_v)
    # shmem_handle.barrier()

    # print(f"rank {rank}: {q.shape = },{k.shape = },{v.shape = },{seq_lens = },{self.window_size = }\n")
    iris_o.copy_(
        flash_attention(
            iris_q,
            iris_k,
            iris_v,
            k_lens=seq_lens,
            window_size=self.window_size,
        )
    )
    x = attn_buffer

    # x = all_to_all(x, scatter_dim=1, gather_dim=2)
    # triton_all_to_all_4D_bf16_backward[(sp_seq_len,1,1)](x,hs,hn//world_size,sp_seq_len*world_size,rank,world_size,iris_o,heap_bases)
    triton_all_to_all_4D_bf16_backward[(sp_seq_len,1,1)](iris_o,hs,hn//world_size,sp_seq_len*world_size,rank,world_size,x,heap_bases)
    # iris_o = iris_o.flatten(2)
    # shmem_handle.barrier()
    x = x.flatten(2)
    x = self.o(x)
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
