import torch
import aiter
import time
from aiter import dtypes
import numpy as np
import argparse

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
    
    # Attention configuration parameters
    parser.add_argument('-dropout_p', type=float, default=0.0,required=False, help='Dropout probability (0.0 for disabled)')
    parser.add_argument('-softmax_scale', type=float, default=1.0/128,required=False, help='Scaling factor for softmax')
    parser.add_argument('-causal', type=lambda x: (str(x).lower() == 'true'), default="false",required=False, help='Use causal attention mask')
    parser.add_argument('-window_size_left', type=int, default=-1,required=False, help='Left window size (-1 for unlimited)')
    parser.add_argument('-window_size_right', type=int, default=-1,required=False, help='Right window size (-1 for unlimited)')
    
    # Advanced options
    parser.add_argument('-bias', type=lambda x: None if str(x).lower() == 'none' else x, required=False,default="none", help='Attention bias tensor or None')
    parser.add_argument('-alibi_slopes', type=lambda x: None if str(x).lower() == 'none' else x, default="none",required=False, help='ALiBi slopes tensor or None')
    parser.add_argument('-return_lse', type=lambda x: (str(x).lower() == 'true'), required=False, default="false",help='Return logsumexp values')
    parser.add_argument('-return_softmax', type=lambda x: (str(x).lower() == 'true'), required=False, default="false",help='Return softmax values')
    
    # Data type selection
    parser.add_argument('-dtype', type=str, choices=['torch.bfloat16', 'torch.float16'] , required=False, default="torch.bfloat16",help='Data type for tensors')
    
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
    
    # Warmup run
    num_warmup_runs = 10
    torch.cuda.synchronize()
    for i in range(num_warmup_runs):
        _ = aiter.ops.mha._flash_attn_forward(
            q, k, v, args.dropout_p, args.softmax_scale, args.causal,
            args.window_size_left, args.window_size_right, args.bias,
            args.alibi_slopes, args.return_lse, args.return_softmax
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
        block_out, block_lse, _, _ = aiter.ops.mha._flash_attn_forward(
            q, k, v, args.dropout_p, args.softmax_scale, args.causal,
            args.window_size_left, args.window_size_right, args.bias,
            args.alibi_slopes, args.return_lse, args.return_softmax
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