#!/bin/bash

# ==================== 监控配置区 ====================
# 生成带时间戳的日志文件名
LOG_FILE="amd_smi_monitor_$(date +%Y%m%d_%H%M%S).log"

# 设置监控超时时间（秒），防止进程无限运行
# 建议设置为测试时长的1.5倍，例如7200秒=2小时
MONITOR_TIMEOUT=7200

# 启动后台监控的函数
start_monitor() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 启动GPU监控，日志: $LOG_FILE"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 超时时间: ${MONITOR_TIMEOUT}秒"
    
    # 使用timeout限制总时长，输出重定向到日志
    timeout ${MONITOR_TIMEOUT}s amd-smi monitor > "$LOG_FILE" &
    MONITOR_PID=$!
}

# 停止监控的清理函数
cleanup_monitor() {
    if [ -n "$MONITOR_PID" ] && kill -0 $MONITOR_PID 2>/dev/null; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] 停止GPU监控 (PID: $MONITOR_PID)"
        kill $MONITOR_PID 2>/dev/null
        wait $MONITOR_PID 2>/dev/null
    fi
}

# 注册trap，确保脚本退出（正常/异常）时自动清理监控进程
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
sudo rocm-smi --setperfdeterminism 1000
docker exec -e ENABLE_TORCH_PROFILER=1 zov_wan2.2_rope_alltoall_fusion_test bash /app/Wan2.2/tests/i2v.sh

# ============= 模型运行后停止监控 =============
sudo rocm-smi -r
cleanup_monitor  # 显式调用清理

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 测试完成，监控日志: $LOG_FILE"
