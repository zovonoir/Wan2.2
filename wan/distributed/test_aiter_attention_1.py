import torch
import aiter
import time
from aiter import dtypes
import numpy as np
import argparse



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


def parse_dtype(dtype_str: str) -> torch.dtype:
    dtype_map = {
        'torch.bfloat16': dtypes.bf16,
        'torch.float16': dtypes.fp16
    }
    return dtype_map[dtype_str]

def parse_args():
    """Parse command-line arguments and return namespace object"""
    parser = argparse.ArgumentParser(description='Flash Attention Forward Performance Test')
    
    # Tensor shape and attention parameters
    parser.add_argument('-batch_size', type=int, default=1,required=False, help='Batch size for input tensors')
    parser.add_argument('-seqlen_q', type=int, default=109120,required=False, help='Sequence length for query tensor')
    parser.add_argument('-nheads', type=int, default=5,required=False, help='Number of attention heads')
    parser.add_argument('-headdim', type=int, default=128,required=False, help='Dimensionality per attention head (query)')
    parser.add_argument('-seqlen_k', type=int, default=109120,required=False, help='Sequence length for key tensor')
    parser.add_argument('-nheads_k', type=int, default=5,required=False, help='Number of attention heads for key')
    parser.add_argument('-head_dim_v', type=int, default=128,required=False, help='Dimensionality per attention head (value)')

    
    return parser.parse_args()

def test_single_run(args):
    """Execute performance test with given parameters"""
    # Set random seed and clear CUDA cache
    torch.random.manual_seed(0)
    torch.cuda.empty_cache()
    
    # Parse data type
    dtype = parse_dtype(args.dtype)
    
    # Create input tensors
    q = torch.randn(args.batch_size, args.seqlen_q, args.nheads, args.headdim, device="cuda", dtype=dtype)
    k = torch.randn(args.batch_size, args.seqlen_k, args.nheads_k, args.headdim, device="cuda", dtype=dtype)
    v = torch.randn(args.batch_size, args.seqlen_k, args.nheads_k, args.head_dim_v, device="cuda", dtype=dtype)
    k_lens=torch.tensor([args.seqlen_q],dtype=torch.long,device="cuda")
    window_size=(-1,-1)

    # Warmup run
    num_warmup_runs = 10
    torch.cuda.synchronize()
    for i in range(num_warmup_runs):
        _ = flash_attention_aiter(
            q, k, v, k_lens=k_lens,window_size=window_size
        )
    torch.cuda.synchronize()

    # Main test loop
    total_time = 0.0
    num_runs = 100
    execution_times = [] 
    print(f"\nStarting benchmark with {num_runs-1} timed runs...")
    for i in range(num_runs):
        # Synchronize GPU for precise timing
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        # Execute Flash Attention operation
        block_out = flash_attention_aiter(
            q, k, v, k_lens=k_lens,window_size=window_size
        )
        
        # Synchronize GPU and record end time
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        
        # Record execution time
        execution_times.append(execution_time)
        total_time += execution_time
        print(f"Run {i}/{num_runs-1}: {execution_time:.6f} seconds")
    
    # Calculate performance statistics
    average_time = total_time / (num_runs)
    min_time = min(execution_times)
    max_time = max(execution_times)
    std_dev = np.std(execution_times) if len(execution_times) > 1 else 0
    
    # Print benchmark results
    print("\n" + "=" * 70)
    print("=" * 70)
    print(f"Average execution time: {average_time:.6f} seconds")
    print(f"Min execution time:    {min_time:.6f} seconds")
    print(f"Max execution time:    {max_time:.6f} seconds")
    print(f"Standard deviation:    {std_dev:.6f} seconds")
    print(f"Total GPU memory used: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    print("=" * 70)
    print(f"Note: First run (warmup) was excluded from statistics")

if __name__ == "__main__":
    args = parse_args()
    test_single_run(args)