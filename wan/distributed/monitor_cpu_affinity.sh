#!/bin/bash

# 实时监控Python进程的CPU绑定情况
# 在运行benchmark的同时，在另一个终端运行此脚本

echo "========================================"
echo "CPU绑核监控脚本"
echo "按Ctrl+C停止监控"
echo "========================================"
echo ""

while true; do
    clear
    echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "========================================"
    echo "Python进程的CPU绑定情况"
    echo "========================================"
    echo ""
    
    # 查找所有benchmark_attention.py进程
    pids=$(pgrep -f "benchmark_attention.py")
    
    if [ -z "$pids" ]; then
        echo "未找到运行中的benchmark_attention.py进程"
    else
        printf "%-8s %-8s %-50s %-10s\n" "PID" "GPU" "CPU Affinity" "NUMA"
        echo "--------------------------------------------------------------------------------"
        
        for pid in $pids; do
            # 获取进程的CPU亲和性
            affinity=$(taskset -cp $pid 2>/dev/null | grep -oP "list: \K.*")
            
            # 获取NUMA节点
            numa=$(cat /proc/$pid/numa_maps 2>/dev/null | head -1 | grep -oP "N\d+" | head -1)
            [ -z "$numa" ] && numa="N/A"
            
            # 尝试从cmdline获取rank参数
            cmdline=$(cat /proc/$pid/cmdline 2>/dev/null | tr '\0' ' ')
            gpu=$(echo "$cmdline" | grep -oP "\-rank\s+\K\d+")
            [ -z "$gpu" ] && gpu="?"
            
            # 简化CPU亲和性显示（如果太长）
            if [ ${#affinity} -gt 40 ]; then
                affinity_short="${affinity:0:37}..."
            else
                affinity_short="$affinity"
            fi
            
            printf "%-8s %-8s %-50s %-10s\n" "$pid" "$gpu" "$affinity_short" "$numa"
        done
        
        echo ""
        echo "========================================"
        echo "CPU使用率（top 16核）"
        echo "========================================"
        mpstat -P ALL 1 1 | head -20 | tail -18
    fi
    
    sleep 2
done

