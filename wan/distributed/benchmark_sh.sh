#!/bin/bash

# 设置GPU时钟频率
yes y | sudo rocm-smi --setsrange 900 1000
mkdir -p /home/jialzhu/clk_900_1000

# 启动后台监控进程，每0.1秒记录一次
LOG_FILE="/home/jialzhu/clk_900_1000/amd_smi_monitor_$(date +%Y%m%d_%H%M%S)_900_1000.log"
echo "开始监控，当前时钟频率 900-1000，日志文件：$LOG_FILE"

(while true; do
    amd-smi dmon --power-usage --temperature --gfx --pcie
    sleep 0.1
done) > "$LOG_FILE" 2>&1 &

MONITOR_PID=$!
echo "监控进程PID: $MONITOR_PID"

trap "echo '停止监控进程...'; kill $MONITOR_PID 2>/dev/null" EXIT
docker exec zov_wan2.2_rope_alltoall_fusion_test bash -c "cd /app/Wan2.2/wan/distributed && torchrun --nnodes=1 --nproc_per_node=1 benchmark.py && sleep 30 && torchrun --nnodes=1 --nproc_per_node=2 benchmark.py  && sleep 30 &&  torchrun --nnodes=1 --nproc_per_node=4 benchmark.py && sleep 30 && torchrun --nnodes=1 --nproc_per_node=8 benchmark.py"

# 在容器内压缩JSON文件
ARCHIVE_NAME="benchmark_results_$(date +%Y%m%d_%H%M%S)_900_1000.7z"
echo "正在压缩结果文件..."
docker exec zov_wan2.2_rope_alltoall_fusion_test bash -c "cd /app/Wan2.2/wan/distributed && 7z a -mx=9 $ARCHIVE_NAME *.json"

# 复制压缩包到宿主机
docker cp zov_wan2.2_rope_alltoall_fusion_test:/app/Wan2.2/wan/distributed/$ARCHIVE_NAME /home/jialzhu/clk_900_1000/
echo "压缩包已复制到 /home/jialzhu/clk_900_1000/$ARCHIVE_NAME"

# 清理容器内的压缩包和原始JSON文件
echo "正在清理容器内的文件..."
docker exec zov_wan2.2_rope_alltoall_fusion_test bash -c "rm -f /app/Wan2.2/wan/distributed/$ARCHIVE_NAME /app/Wan2.2/wan/distributed/*.json"
echo "容器内文件已清理"

sudo rocm-smi -r
echo "测试完成，日志已保存到 $LOG_FILE"





# 设置GPU时钟频率
yes y | sudo rocm-smi --setsrange 2300 2400
mkdir -p /home/jialzhu/clk_2300_2400

# 启动后台监控进程，每0.1秒记录一次
LOG_FILE="/home/jialzhu/clk_2300_2400/amd_smi_monitor_$(date +%Y%m%d_%H%M%S)_2300_2400.log"
echo "开始监控，当前时钟频率 2300-2400，日志文件：$LOG_FILE"

(while true; do
    amd-smi dmon --power-usage --temperature --gfx --pcie
    sleep 0.1
done) > "$LOG_FILE" 2>&1 &

MONITOR_PID=$!
echo "监控进程PID: $MONITOR_PID"

trap "echo '停止监控进程...'; kill $MONITOR_PID 2>/dev/null" EXIT
docker exec zov_wan2.2_rope_alltoall_fusion_test bash -c "cd /app/Wan2.2/wan/distributed && torchrun --nnodes=1 --nproc_per_node=1 benchmark.py && sleep 30 && torchrun --nnodes=1 --nproc_per_node=2 benchmark.py  && sleep 30 &&  torchrun --nnodes=1 --nproc_per_node=4 benchmark.py && sleep 30 && torchrun --nnodes=1 --nproc_per_node=8 benchmark.py"

# 在容器内压缩JSON文件
ARCHIVE_NAME="benchmark_results_$(date +%Y%m%d_%H%M%S)_2300_2400.7z"
echo "正在压缩结果文件..."
docker exec zov_wan2.2_rope_alltoall_fusion_test bash -c "cd /app/Wan2.2/wan/distributed && 7z a -mx=9 $ARCHIVE_NAME *.json"

# 复制压缩包到宿主机
docker cp zov_wan2.2_rope_alltoall_fusion_test:/app/Wan2.2/wan/distributed/$ARCHIVE_NAME /home/jialzhu/clk_2300_2400/
echo "压缩包已复制到 /home/jialzhu/clk_2300_2400/$ARCHIVE_NAME"

# 清理容器内的压缩包和原始JSON文件
echo "正在清理容器内的文件..."
docker exec zov_wan2.2_rope_alltoall_fusion_test bash -c "rm -f /app/Wan2.2/wan/distributed/$ARCHIVE_NAME /app/Wan2.2/wan/distributed/*.json"
echo "容器内文件已清理"

sudo rocm-smi -r
echo "测试完成，日志已保存到 $LOG_FILE"

