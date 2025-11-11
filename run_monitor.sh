#!/bin/bash

# ==================== 监控配置区 ====================
# 生成带时间戳的日志文件名
LOG_FILE="amd_smi_monitor_$(date +%Y%m%d_%H%M%S).log"
SAMPLE_INTERVAL=0.5  # 采样间隔0.5秒
MONITOR_TIMEOUT=7200  # 超时保护（2小时）

# 启动后台监控函数
start_monitor() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 启动GPU监控，日志: $LOG_FILE，采样间隔: ${SAMPLE_INTERVAL}秒"
    
    # 使用while循环实现持续监控
    {
        # 记录起始时间
        START_TIME=$(date +%s)
        SAMPLE_COUNT=0
        
        while true; do
            # 检查是否超时
            CUR_TIME=$(date +%s)
            if [ $((CUR_TIME - START_TIME)) -gt $MONITOR_TIMEOUT ]; then
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] 监控超时，自动退出" >&2
                break
            fi
            
            # 记录时间戳和采样序号
            echo "=== SAMPLE_$SAMPLE_COUNT | $(date '+%Y-%m-%d %H:%M:%S.%N') ===" >> "$LOG_FILE"
            
            # 执行单次监控并追加到日志
            amd-smi monitor >> "$LOG_FILE" 2>&1
            
            # 如果命令失败，记录错误
            if [ $? -ne 0 ]; then
                echo "[ERROR] amd-smi monitor 执行失败" >> "$LOG_FILE"
            fi
            
            # 间隔0.5秒
            sleep "$SAMPLE_INTERVAL"
            SAMPLE_COUNT=$((SAMPLE_COUNT + 1))
        done
    } &
    MONITOR_PID=$!
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 监控进程PID: $MONITOR_PID"
}

# 停止监控的清理函数
cleanup_monitor() {
    if [ -n "$MONITOR_PID" ]; then
        if kill -0 $MONITOR_PID 2>/dev/null; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] 停止GPU监控 (PID: $MONITOR_PID)"
            kill $MONITOR_PID 2>/dev/null
            wait $MONITOR_PID 2>/dev/null
        fi
    fi
    
    # 检查日志文件是否存在
    if [ -f "$LOG_FILE" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] 监控日志已保存: $LOG_FILE"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] 日志大小: $(du -h "$LOG_FILE" | cut -f1)"
    fi
}

# 注册trap，确保脚本退出时自动清理监控进程
trap cleanup_monitor EXIT INT TERM

# ==================== 原有测试逻辑 ====================
docker rm -f zov_wan2.2_rope_alltoall_fusion_test
docker rmi wan2.2_rocm7.0.2_pytorch_2.10.0.dev20251023_iris_0dfc460_2

docker build -f dockerfile_rocm_7.0.2 -t wan2.2_rocm7.0.2_pytorch_2.10.0.dev20251023_iris_0dfc460_2 .

docker run -d \
    --network=host \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --shm-size=16G \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v /home/jialzhu:/home/jialzhu \
    --name zov_wan2.2_rope_alltoall_fusion_test \
    -t wan2.2_rocm7.0.2_pytorch_2.10.0.dev20251023_iris_0dfc460_2

docker restart zov_wan2.2_rope_alltoall_fusion_test
docker exec zov_wan2.2_rope_alltoall_fusion_test git -C /app/Wan2.2 pull

# ============= 在模型运行前启动监控 =============
start_monitor

# ============= 模型运行阶段 =============
sudo rocm-smi --setperfdeterminism 2400
docker exec -e ENABLE_TORCH_PROFILER=1 zov_wan2.2_rope_alltoall_fusion_test bash /app/Wan2.2/tests/i2v.sh

# ============= 模型运行后停止监控 =============
sudo rocm-smi -r
cleanup_monitor  # 显式调用清理

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 测试完成"