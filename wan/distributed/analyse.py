#!/usr/bin/env python3
"""
分析GPU kernel运行时间的脚本
扫描当前目录下所有JSON文件，统计指定kernel的耗时
"""

import json
import glob
import os
from pathlib import Path
from typing import List, Dict, Tuple
from multiprocessing import Pool, cpu_count
from collections import defaultdict

# ========== 配置区域 ==========
# 修改此变量来分析不同的kernel
KERNEL_NAME = "alltoallbackward"
# ==============================


def process_json_file(filepath: str) -> Tuple[str, List[float]]:
    """
    处理单个JSON文件，提取指定kernel的运行时间
    
    Args:
        filepath: JSON文件路径
    
    Returns:
        (文件名, 耗时列表)
    """
    durations = []
    filename = os.path.basename(filepath)
    
    try:
        # 对于超大文件，使用流式读取会更好，但标准json.load在大多数情况下已经足够快
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # 处理不同的JSON结构
        events = []
        if isinstance(data, dict):
            # 如果是对象，尝试获取 traceEvents 字段
            events = data.get("traceEvents", [])
        elif isinstance(data, list):
            # 如果直接是数组
            events = data
        
        # 遍历所有事件，筛选出目标kernel
        for event in events:
            if isinstance(event, dict):
                # 检查 name 字段
                if event.get("cat") == "kernel" and event.get("name") == KERNEL_NAME:
                    dur = event.get("dur")
                    if dur is not None:
                        durations.append(float(dur))
                # 也检查 args.kernel 字段
                elif event.get("cat") == "kernel":
                    args = event.get("args", {})
                    if isinstance(args, dict) and args.get("kernel") == KERNEL_NAME:
                        dur = event.get("dur")
                        if dur is not None:
                            durations.append(float(dur))
        
        print(f"处理完成: {filename} - 找到 {len(durations)} 个 {KERNEL_NAME} kernel")
        
    except json.JSONDecodeError as e:
        print(f"警告: 无法解析 {filename}: {e}")
    except Exception as e:
        print(f"错误: 处理 {filename} 时出错: {e}")
    
    return filename, durations


def analyze_kernel_timing():
    """主函数：扫描JSON文件并统计kernel耗时"""
    
    # 获取当前脚本所在目录
    script_dir = Path(__file__).parent
    
    # 查找所有JSON文件
    json_files = glob.glob(str(script_dir / "*.json"))
    
    if not json_files:
        print(f"错误: 在 {script_dir} 目录下未找到任何JSON文件")
        return
    
    print(f"找到 {len(json_files)} 个JSON文件")
    print(f"分析kernel: {KERNEL_NAME}")
    print("=" * 60)
    
    # 使用多进程并行处理文件
    num_processes = min(cpu_count(), len(json_files))
    
    with Pool(processes=num_processes) as pool:
        results = pool.map(process_json_file, json_files)
    
    # 统计结果
    file_stats = {}
    total_duration = 0.0
    total_count = 0
    all_durations = []  # 收集所有文件的所有耗时
    
    print("\n" + "=" * 60)
    print("统计结果:")
    print("=" * 60)
    
    for filename, durations in results:
        if durations:
            count = len(durations)
            total = sum(durations)
            average = total / count
            max_dur = max(durations)
            min_dur = min(durations)
            
            file_stats[filename] = {
                'count': count,
                'total': total,
                'average': average,
                'max': max_dur,
                'min': min_dur
            }
            
            total_duration += total
            total_count += count
            all_durations.extend(durations)
            
            print(f"\n文件: {filename}")
            print(f"  - {KERNEL_NAME} 数量: {count}")
            print(f"  - 总耗时: {total:.2f} μs ({total/1000:.2f} ms)")
            print(f"  - 平均耗时: {average:.2f} μs ({average/1000:.4f} ms)")
            print(f"  - 最长耗时: {max_dur:.2f} μs ({max_dur/1000:.4f} ms)")
            print(f"  - 最短耗时: {min_dur:.2f} μs ({min_dur/1000:.4f} ms)")
    
    # 计算总体统计
    if total_count > 0:
        global_average = total_duration / total_count
        global_max = max(all_durations)
        global_min = min(all_durations)
        
        print("\n" + "=" * 60)
        print("总体统计:")
        print("=" * 60)
        print(f"总文件数: {len(file_stats)}")
        print(f"总 {KERNEL_NAME} 数量: {total_count}")
        print(f"所有文件总耗时: {total_duration:.2f} μs ({total_duration/1000:.2f} ms)")
        print(f"所有文件平均耗时: {global_average:.2f} μs ({global_average/1000:.4f} ms)")
        print(f"所有文件最长耗时: {global_max:.2f} μs ({global_max/1000:.4f} ms)")
        print(f"所有文件最短耗时: {global_min:.2f} μs ({global_min/1000:.4f} ms)")
        
        # 计算每个文件的平均耗时的平均值（如果需要的话）
        file_averages = [stats['average'] for stats in file_stats.values()]
        avg_of_averages = sum(file_averages) / len(file_averages)
        print(f"各文件平均耗时的平均值: {avg_of_averages:.2f} μs ({avg_of_averages/1000:.4f} ms)")
        
    else:
        print(f"\n警告: 在所有JSON文件中未找到任何 {KERNEL_NAME} kernel")


if __name__ == "__main__":
    analyze_kernel_timing()

