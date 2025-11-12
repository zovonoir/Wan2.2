#!/bin/bash

# 对比测试：无绑核 vs 有绑核
# 这个脚本会运行两次测试并收集性能数据

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="./benchmark_logs_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "========================================"
echo "性能对比测试"
echo "时间戳: $TIMESTAMP"
echo "日志目录: $LOG_DIR"
echo "========================================"

# ==========================================
# 测试1: 不绑核（基准测试）
# ==========================================
echo ""
echo "========================================"
echo "测试1: 不绑核（基准测试）"
echo "========================================"
date

# 清理旧的trace文件
rm -f flash_attention_pressure_testing_rank_*.json

# 启动GPU监控
(while true; do
    date +%s.%N
    rocm-smi | head -15
    sleep 0.1
done) > "$LOG_DIR/gpu_monitor_no_bind.log" 2>&1 &
MONITOR_PID=$!
echo "GPU监控已启动 (PID: $MONITOR_PID)"

# 运行测试（不绑核）
for i in {0..7}; do
    CUDA_VISIBLE_DEVICES=$i python benchmark_attention.py -rank $i &
done

# 等待所有进程完成
wait

# 停止监控
kill $MONITOR_PID 2>/dev/null
echo "GPU监控已停止"

# 收集trace文件
mv flash_attention_pressure_testing_rank_*.json "$LOG_DIR/" 2>/dev/null
echo "Trace文件已保存到 $LOG_DIR/"

echo "测试1完成"
date
sleep 5

# ==========================================
# 测试2: 带CPU绑核和NUMA优化
# ==========================================
echo ""
echo "========================================"
echo "测试2: 带CPU绑核和NUMA优化"
echo "========================================"
date

# 设置环境变量
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# 清理旧的trace文件
rm -f flash_attention_pressure_testing_rank_*.json

# 启动GPU监控
(while true; do
    date +%s.%N
    rocm-smi | head -15
    sleep 0.1
done) > "$LOG_DIR/gpu_monitor_with_bind.log" 2>&1 &
MONITOR_PID=$!
echo "GPU监控已启动 (PID: $MONITOR_PID)"

# 运行测试（带绑核）
# GPU 0-3使用NUMA node0
CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=0 --membind=0 taskset -c 0-15 python benchmark_attention.py -rank 0 &
CUDA_VISIBLE_DEVICES=1 numactl --cpunodebind=0 --membind=0 taskset -c 16-31 python benchmark_attention.py -rank 1 &
CUDA_VISIBLE_DEVICES=2 numactl --cpunodebind=0 --membind=0 taskset -c 32-47 python benchmark_attention.py -rank 2 &
CUDA_VISIBLE_DEVICES=3 numactl --cpunodebind=0 --membind=0 taskset -c 48-63 python benchmark_attention.py -rank 3 &

# GPU 4-7使用NUMA node1
CUDA_VISIBLE_DEVICES=4 numactl --cpunodebind=1 --membind=1 taskset -c 64-79 python benchmark_attention.py -rank 4 &
CUDA_VISIBLE_DEVICES=5 numactl --cpunodebind=1 --membind=1 taskset -c 80-95 python benchmark_attention.py -rank 5 &
CUDA_VISIBLE_DEVICES=6 numactl --cpunodebind=1 --membind=1 taskset -c 96-111 python benchmark_attention.py -rank 6 &
CUDA_VISIBLE_DEVICES=7 numactl --cpunodebind=1 --membind=1 taskset -c 112-127 python benchmark_attention.py -rank 7 &

# 等待所有进程完成
wait

# 停止监控
kill $MONITOR_PID 2>/dev/null
echo "GPU监控已停止"

# 收集trace文件并重命名
for f in flash_attention_pressure_testing_rank_*.json; do
    if [ -f "$f" ]; then
        # 在文件名中添加"_with_bind"标记
        new_name=$(echo "$f" | sed 's/.json/_with_bind.json/')
        mv "$f" "$LOG_DIR/$new_name"
    fi
done
echo "Trace文件已保存到 $LOG_DIR/"

echo "测试2完成"
date

# ==========================================
# 生成对比分析脚本
# ==========================================
cat > "$LOG_DIR/analyze.py" << 'EOF'
#!/usr/bin/env python3
"""
分析对比测试结果
"""
import json
import os
from pathlib import Path

def analyze_trace(trace_file):
    """分析单个trace文件"""
    with open(trace_file, 'r') as f:
        data = json.load(f)
    
    # 提取算子执行时间
    events = data.get('traceEvents', [])
    
    kernel_times = []
    for event in events:
        if event.get('cat') == 'kernel' or event.get('cat') == 'Kernel':
            dur = event.get('dur', 0)
            if dur > 0:
                kernel_times.append(dur / 1000.0)  # 转换为ms
    
    if kernel_times:
        total_time = sum(kernel_times)
        avg_time = total_time / len(kernel_times)
        return {
            'count': len(kernel_times),
            'total_ms': total_time,
            'avg_ms': avg_time,
            'min_ms': min(kernel_times),
            'max_ms': max(kernel_times)
        }
    return None

def main():
    """主函数"""
    current_dir = Path('.')
    
    print("=" * 60)
    print("性能对比分析")
    print("=" * 60)
    
    # 分析无绑核的结果
    print("\n【测试1：无绑核】")
    no_bind_files = sorted(current_dir.glob('flash_attention_pressure_testing_rank_*[!bind].json'))
    no_bind_results = {}
    
    for f in no_bind_files:
        rank = int(f.name.split('_rank_')[1].split('_')[0])
        result = analyze_trace(f)
        if result:
            no_bind_results[rank] = result
            print(f"  Rank {rank}: 平均 {result['avg_ms']:.2f}ms, "
                  f"总时间 {result['total_ms']:.2f}ms, "
                  f"算子数 {result['count']}")
    
    # 分析有绑核的结果
    print("\n【测试2：有绑核】")
    with_bind_files = sorted(current_dir.glob('flash_attention_pressure_testing_rank_*_with_bind.json'))
    with_bind_results = {}
    
    for f in with_bind_files:
        rank = int(f.name.split('_rank_')[1].split('_')[0])
        result = analyze_trace(f)
        if result:
            with_bind_results[rank] = result
            print(f"  Rank {rank}: 平均 {result['avg_ms']:.2f}ms, "
                  f"总时间 {result['total_ms']:.2f}ms, "
                  f"算子数 {result['count']}")
    
    # 对比分析
    print("\n【性能对比】")
    if no_bind_results and with_bind_results:
        print(f"{'Rank':<8} {'无绑核(ms)':<15} {'有绑核(ms)':<15} {'改善率':<10}")
        print("-" * 60)
        
        for rank in sorted(set(no_bind_results.keys()) & set(with_bind_results.keys())):
            no_bind_avg = no_bind_results[rank]['avg_ms']
            with_bind_avg = with_bind_results[rank]['avg_ms']
            improvement = (no_bind_avg - with_bind_avg) / no_bind_avg * 100
            
            print(f"Rank {rank:<3} {no_bind_avg:>12.2f}    {with_bind_avg:>12.2f}    "
                  f"{improvement:>+6.1f}%")
        
        # 总体统计
        all_no_bind_avg = sum(r['avg_ms'] for r in no_bind_results.values()) / len(no_bind_results)
        all_with_bind_avg = sum(r['avg_ms'] for r in with_bind_results.values()) / len(with_bind_results)
        overall_improvement = (all_no_bind_avg - all_with_bind_avg) / all_no_bind_avg * 100
        
        print("-" * 60)
        print(f"{'平均':<8} {all_no_bind_avg:>12.2f}    {all_with_bind_avg:>12.2f}    "
              f"{overall_improvement:>+6.1f}%")
        
        if overall_improvement > 0:
            print(f"\n✓ CPU绑核优化提升了 {overall_improvement:.1f}% 的性能！")
        elif overall_improvement < -5:
            print(f"\n✗ CPU绑核反而降低了 {-overall_improvement:.1f}% 的性能")
        else:
            print(f"\n≈ CPU绑核对性能影响不大（{overall_improvement:.1f}%）")
    
    print("\n" + "=" * 60)

if __name__ == '__main__':
    main()
EOF

chmod +x "$LOG_DIR/analyze.py"

echo ""
echo "========================================"
echo "所有测试完成！"
echo "========================================"
echo ""
echo "日志文件保存在: $LOG_DIR/"
echo ""
echo "要查看分析结果，请运行:"
echo "  cd $LOG_DIR && python3 analyze.py"
echo ""
echo "要查看GPU监控日志:"
echo "  无绑核: $LOG_DIR/gpu_monitor_no_bind.log"
echo "  有绑核: $LOG_DIR/gpu_monitor_with_bind.log"
echo ""

