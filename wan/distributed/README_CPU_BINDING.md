# CPU绑核优化测试指南

## 问题背景

在多GPU并发测试中发现，随着进程数量增加，算子执行速度变慢。通过监控发现GPU利用率很低，大部分时间处于空闲状态，这表明问题可能出在CPU端的调度和竞争上。

## 服务器配置

- **CPU**: 2 × AMD EPYC 9575F (每个64核心/128线程)
- **GPU**: 8 × AMD GPU
  - GPU 0-3: NUMA node 0
  - GPU 4-7: NUMA node 1
- **电源**: 4 × 6600W PSU (总容量26,400W)

## 优化策略

### 1. CPU线程限制
设置 `torch.set_num_threads(1)` 避免CPU线程超额订阅

### 2. CPU绑核
将每个进程绑定到特定的CPU核心，避免进程间争用：
- GPU 0: CPU 0-15 (NUMA 0)
- GPU 1: CPU 16-31 (NUMA 0)
- GPU 2: CPU 32-47 (NUMA 0)
- GPU 3: CPU 48-63 (NUMA 0)
- GPU 4: CPU 64-79 (NUMA 1)
- GPU 5: CPU 80-95 (NUMA 1)
- GPU 6: CPU 96-111 (NUMA 1)
- GPU 7: CPU 112-127 (NUMA 1)

### 3. NUMA亲和性
确保GPU和对应的CPU/内存在同一个NUMA节点上，减少跨NUMA访问延迟

## 使用方法

### 方法1：快速测试（仅绑核）

```bash
# 给脚本添加执行权限
chmod +x benchmark_sh2.sh

# 运行测试
./benchmark_sh2.sh
```

这个脚本会启动8个进程，每个进程：
- 使用独立的GPU
- 绑定到特定的CPU核心
- 绑定到正确的NUMA节点
- 限制使用单线程

### 方法2：完整对比测试

```bash
# 给脚本添加执行权限
chmod +x benchmark_compare.sh

# 运行对比测试（需要较长时间，会运行两轮测试）
./benchmark_compare.sh
```

这个脚本会：
1. 运行无绑核的基准测试
2. 运行有绑核的优化测试
3. 收集GPU监控数据
4. 自动生成性能对比分析

测试完成后，运行分析脚本：

```bash
cd benchmark_logs_<timestamp>
python3 analyze.py
```

### 方法3：实时监控

在运行测试的同时，可以在另一个终端监控CPU绑定情况：

```bash
# 给脚本添加执行权限
chmod +x monitor_cpu_affinity.sh

# 启动监控
./monitor_cpu_affinity.sh
```

按 `Ctrl+C` 停止监控。

## 手动验证绑核

### 查看进程的CPU亲和性

```bash
# 查找benchmark进程
pgrep -f benchmark_attention.py

# 查看特定进程的CPU绑定
taskset -cp <PID>
```

### 查看NUMA信息

```bash
# 查看系统NUMA配置
numactl --hardware

# 查看进程的NUMA统计
numastat -p <PID>
```

### 查看GPU的NUMA分布

```bash
rocm-smi --showtoponuma
```

## 预期效果

### 如果绑核有效

- **GPU利用率提升**: 从接近0%提升到80-100%
- **算子执行时间缩短**: 可能有20-50%的改善
- **功耗增加**: GPU功耗从300W增加到1000W+（说明真正在计算）

### 如果绑核无效

如果绑核后性能没有明显改善，可能的原因：
1. **ROCm驱动调度问题**: 多进程kernel调度效率低
2. **GPU硬件调度问题**: 硬件调度器在多进程场景下表现不佳
3. **PCIe命令通道瓶颈**: 命令提交通道拥塞
4. **隐式同步**: 代码中存在未发现的同步点

## 进一步调试

### 1. 使用rocprof分析

```bash
# 对单个进程进行详细profiling
rocprof --hip-trace --stats python benchmark_attention.py -rank 0
```

### 2. 查看kernel launch延迟

在 `benchmark_attention.py` 中添加详细的timing：

```python
import time

start = time.perf_counter()
torch.cuda.synchronize()
kernel_start = time.perf_counter()

# 执行算子
flash_attention_aiter(...)

torch.cuda.synchronize()
kernel_end = time.perf_counter()

print(f"Launch overhead: {(kernel_start - start)*1000:.2f}ms")
print(f"Kernel time: {(kernel_end - kernel_start)*1000:.2f}ms")
```

### 3. 禁用分布式初始化测试

修改 `benchmark_attention.py`，注释掉 `dist.init_process_group()`，看是否有改善。

## 文件说明

- `benchmark_sh2.sh`: 带NUMA和CPU绑核的启动脚本
- `benchmark_compare.sh`: 对比测试脚本（无绑核 vs 有绑核）
- `monitor_cpu_affinity.sh`: CPU绑定实时监控脚本
- `benchmark_attention.py`: 已优化的测试脚本（支持单线程和设备设置）

## 环境变量说明

脚本中设置的环境变量：

- `OMP_NUM_THREADS=1`: OpenMP使用单线程
- `MKL_NUM_THREADS=1`: Intel MKL使用单线程
- `OPENBLAS_NUM_THREADS=1`: OpenBLAS使用单线程
- `CUDA_VISIBLE_DEVICES=N`: 限制进程只能看到特定GPU

## 故障排查

### 脚本执行失败

```bash
# 检查numactl是否安装
which numactl

# 如果未安装
sudo apt-get install numactl

# 检查taskset是否可用
which taskset
```

### 进程没有绑定到正确的CPU

```bash
# 查看进程的CPU亲和性
ps aux | grep benchmark_attention
taskset -cp <PID>

# 应该看到类似 "current affinity list: 0-15" 的输出
```

### GPU使用率仍然很低

如果绑核后GPU使用率仍然很低，说明问题可能在更底层：
1. 检查ROCm版本是否最新
2. 尝试降低进程数量（4个、2个）看是否改善
3. 联系AMD技术支持，可能是ROCm的已知问题

## 参考资料

- [PyTorch Multiprocessing Best Practices](https://pytorch.org/docs/stable/notes/multiprocessing.html)
- [NUMA架构和优化](https://www.kernel.org/doc/html/latest/vm/numa.html)
- [ROCm System Management](https://rocm.docs.amd.com/en/latest/understand/gpu_arch.html)

