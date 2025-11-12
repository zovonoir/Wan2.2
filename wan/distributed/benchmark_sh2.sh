#!/bin/bash

# 设置环境变量以减少线程竞争
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "========================================"
echo "开始测试 - 带NUMA和CPU绑核优化"
echo "GPU 0-3 绑定到 NUMA node 0"
echo "GPU 4-7 绑定到 NUMA node 1"
echo "========================================"
date

# GPU 0-3使用NUMA node0的物理核心 (0-63)
# 每个GPU分配16个物理核
CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=0 --membind=0 taskset -c 0-15 python benchmark_attention.py -rank 0 &
PID0=$!
echo "GPU 0 启动 (PID: $PID0, CPU: 0-15, NUMA: 0)"

CUDA_VISIBLE_DEVICES=1 numactl --cpunodebind=0 --membind=0 taskset -c 16-31 python benchmark_attention.py -rank 1 &
PID1=$!
echo "GPU 1 启动 (PID: $PID1, CPU: 16-31, NUMA: 0)"

CUDA_VISIBLE_DEVICES=2 numactl --cpunodebind=0 --membind=0 taskset -c 32-47 python benchmark_attention.py -rank 2 &
PID2=$!
echo "GPU 2 启动 (PID: $PID2, CPU: 32-47, NUMA: 0)"

CUDA_VISIBLE_DEVICES=3 numactl --cpunodebind=0 --membind=0 taskset -c 48-63 python benchmark_attention.py -rank 3 &
PID3=$!
echo "GPU 3 启动 (PID: $PID3, CPU: 48-63, NUMA: 0)"

# GPU 4-7使用NUMA node1的物理核心 (64-127)
CUDA_VISIBLE_DEVICES=4 numactl --cpunodebind=1 --membind=1 taskset -c 64-79 python benchmark_attention.py -rank 4 &
PID4=$!
echo "GPU 4 启动 (PID: $PID4, CPU: 64-79, NUMA: 1)"

CUDA_VISIBLE_DEVICES=5 numactl --cpunodebind=1 --membind=1 taskset -c 80-95 python benchmark_attention.py -rank 5 &
PID5=$!
echo "GPU 5 启动 (PID: $PID5, CPU: 80-95, NUMA: 1)"

CUDA_VISIBLE_DEVICES=6 numactl --cpunodebind=1 --membind=1 taskset -c 96-111 python benchmark_attention.py -rank 6 &
PID6=$!
echo "GPU 6 启动 (PID: $PID6, CPU: 96-111, NUMA: 1)"

CUDA_VISIBLE_DEVICES=7 numactl --cpunodebind=1 --membind=1 taskset -c 112-127 python benchmark_attention.py -rank 7 &
PID7=$!
echo "GPU 7 启动 (PID: $PID7, CPU: 112-127, NUMA: 1)"

echo "========================================"
echo "所有进程已启动"
echo "等待所有进程完成..."
echo "========================================"

# 等待所有进程完成
wait $PID0 $PID1 $PID2 $PID3 $PID4 $PID5 $PID6 $PID7

echo "========================================"
echo "测试完成"
date
echo "========================================"
