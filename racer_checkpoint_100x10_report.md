# RACER Megatron 100x10 Checkpoint 时间报告

## 运行配置

- 脚本：`/workspace/Megatron-LM-FT/train_gpt2_1_5b.sh`
- 模型规模：脚本中的 GPT2 1.5B 配置
- GPU：`CUDA_VISIBLE_DEVICES=1,2,3,4,5`；4 个训练 rank；spare rank/device index 为 `4`
- 迭代数：`TRAIN_ITERS=100`；checkpoint 间隔：`SAVE_INTERVAL=10`
- RACER：`RACER_ENABLE=1`，`RACER_DISTRIBUTED_STORE=1`，`RACER_ASYNC_OFFLOAD=0`，`RACER_RETAIN_CHECKPOINTS=1`
- 验证方式：`RACER_VERIFY_LOAD_AFTER_SAVE=1`；每次 checkpoint 保存后立即加载验证
- 日志：`/workspace/logs/gpt2_1.5b_racer_100x10_retain1/run.log`
- shell `time` 记录的总耗时：`1m47.745s`

## 指标定义

- `Store total`：RACER 适配层的 tensor-tree store 总路径耗时，即从 状态字典中的 tensor leaves 写入分布式 RACER 内存 store 的时间。
- `Store 元数据`：构建并发送 tensor tree 元数据的时间。
- `Store tensor_view`：为 tensor leaves 创建 byte view 的时间。这不是 pickle 序列化，只是 tensor 到 byte view 的准备。
- `Store racer_calls`：RACER distributed store 调用和 NCCL 传输路径上的时间。
- `Blocking save_checkpoint() total`：最慢 rank 在 Megatron `save_checkpoint()` 函数内部停留的 wall time。
- `Blocking train-loop timer`：Megatron 训练循环中包住保存调用的 `save-checkpoint` timer；这是训练进程保存阶段的主要阻塞时间。
- `Load adapter total`：RACER 适配层的 tensor-tree load 总路径耗时，即从分布式 RACER 内存 store 恢复为 checkpoint 状态字典的时间。
- `Full load_checkpoint verify`：每次保存后用于正确性验证的完整 Megatron `load_checkpoint()` 调用耗时。

## 汇总

| 指标 | 次数 | 平均 ms | 中位数 ms | 最小 ms | 最大 ms | 标准差 ms |
|---|---:|---:|---:|---:|---:|---:|
| Store 总时间 | 10 | 1707.57 | 1437.47 | 1388.69 | 2946.79 | 467.58 |
| Store 元数据 | 10 | 3.48 | 3.49 | 3.34 | 3.63 | 0.11 |
| Store tensor_view 准备 | 10 | 11.73 | 11.68 | 11.54 | 12.20 | 0.20 |
| Store RACER 调用 | 10 | 1695.48 | 1425.26 | 1376.74 | 2934.57 | 467.60 |
| `save_checkpoint()` 阻塞总时间 | 10 | 1844.46 | 1587.58 | 1471.48 | 3104.84 | 475.91 |
| 训练循环 timer 阻塞 | 10 | 1851.48 | 1592.98 | 1485.18 | 3119.43 | 478.22 |
| Load 适配层总时间 | 10 | 279.40 | 277.55 | 266.90 | 307.86 | 11.29 |
| Load RACER 拉取 | 10 | 265.65 | 263.88 | 253.27 | 294.33 | 11.21 |
| Load tensor 物化 | 10 | 15.61 | 14.99 | 14.57 | 18.36 | 1.34 |
| Load tree 解码 | 10 | 0.98 | 0.96 | 0.77 | 1.23 | 0.14 |
| Load tensor 重建 | 10 | 16.60 | 16.01 | 15.55 | 19.34 | 1.34 |
| 完整 `load_checkpoint()` 验证 | 10 | 307.25 | 303.96 | 290.80 | 349.74 | 15.02 |

## 每个 checkpoint 的明细

| 迭代 | Store ms | 元数据 | Tensor view | RACER 调用 | `save_checkpoint()` 阻塞 | 训练循环阻塞 | Load 适配层 | 拉取 | 物化 | 解码 | 完整 `load_checkpoint()` | Payload GiB | 本地 GiB | Tensor leaves |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 1445.78 | 3.60 | 11.78 | 1433.36 | 1471.48 | 1485.18 | 307.86 | 294.33 | 18.36 | 0.77 | 349.74 | 21.36 | 5.88 | 654 |
| 20 | 1675.51 | 3.39 | 11.74 | 1662.95 | 1840.31 | 1866.59 | 273.55 | 259.88 | 14.94 | 0.88 | 307.54 | 21.36 | 5.88 | 654 |
| 30 | 1423.49 | 3.34 | 11.96 | 1411.19 | 1600.77 | 1610.85 | 269.45 | 256.22 | 15.32 | 0.91 | 296.83 | 21.36 | 5.88 | 654 |
| 40 | 2946.79 | 3.35 | 11.64 | 2934.57 | 3104.84 | 3119.43 | 277.67 | 264.08 | 14.81 | 0.98 | 301.68 | 21.36 | 5.88 | 654 |
| 50 | 1398.25 | 3.46 | 11.59 | 1386.23 | 1559.83 | 1560.43 | 278.25 | 264.19 | 14.91 | 1.05 | 303.90 | 21.36 | 5.88 | 654 |
| 60 | 1419.67 | 3.63 | 11.63 | 1407.66 | 1561.34 | 1561.97 | 266.90 | 253.27 | 14.57 | 1.23 | 290.80 | 21.36 | 5.88 | 654 |
| 70 | 1429.16 | 3.56 | 11.72 | 1417.15 | 1574.39 | 1575.12 | 277.84 | 264.11 | 18.13 | 1.21 | 306.49 | 21.36 | 5.88 | 654 |
| 80 | 1388.69 | 3.52 | 11.54 | 1376.74 | 1518.88 | 1519.54 | 277.42 | 263.69 | 15.34 | 0.93 | 304.03 | 21.36 | 5.88 | 654 |
| 90 | 1959.47 | 3.36 | 11.54 | 1947.47 | 2091.11 | 2092.74 | 274.03 | 260.25 | 14.69 | 0.86 | 303.11 | 21.36 | 5.88 | 654 |
| 100 | 1988.90 | 3.62 | 12.20 | 1977.49 | 2121.67 | 2122.93 | 291.00 | 276.53 | 15.04 | 1.01 | 308.34 | 21.36 | 5.88 | 654 |

## 健康检查

- 已完成的 checkpoint 迭代：`[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]`
- 缺失的预期 checkpoint 迭代：`[]`
- 日志中的错误标记：`[]`
- 保留策略清理事件：`9`；累计清理 checkpoint：`9`；累计清理 leaf state：`5886`
- 每次保存的 checkpoint payload 约为 `21.36 GiB`，跨所有训练 rank 统计；最大单 rank 本地 payload 约为 `5.88 GiB`。
