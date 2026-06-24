# RACER Megatron Checkpoint 时间记录

环境：`CUDA_VISIBLE_DEVICES=1,2,3,4,5`，使用 Megatron GPT2 1.5B 启动脚本 `train_gpt2_1_5b.sh`，4 个训练 rank 加 1 张 spare GPU。

当前实现：分布式 tensor-tree RACER checkpoint。默认 RACER 路径会把 状态字典 中的 tensor leaf 直接交给 `racer.distributed_store` 保存；不会再构造 rank 级别的大 byte payload，也不会通过 Megatron 的 CPU `dp_zero` checkpoint 路径临时搬运分布式优化器状态。

## 测量结果

| 场景 | 命令形态 | 保存时间 | RACER store | 加载时间 | 说明 |
| --- | --- | ---: | ---: | ---: | --- |
| tensor-tree / no-copy optimizer 修复前的旧 adapter 路径 | `TRAIN_ITERS=1 SAVE_INTERVAL=1 RACER_DISTRIBUTED_STORE=1` | 21.94 s | 3.10 s | 该 save-only 命令未测试加载 | 包含 CPU 分布式优化器 staging 和 rank payload 处理 |
| 当前 tensor-tree 路径，仅保存 | `TRAIN_ITERS=1 SAVE_INTERVAL=1 RACER_DISTRIBUTED_STORE=1` | 1.59 s | 1.56 s | n/a | 654 个 tensor leaf，总 tensor 字节数 22.93 GB |
| 当前 tensor-tree 路径，恢复验证 iteration 1 | `TRAIN_ITERS=2 SAVE_INTERVAL=1 RACER_VERIFY_LOAD_AFTER_SAVE=1` | 1.48 s | 1.45 s | 完整 `load_checkpoint()` 为 0.351 s | 之后继续训练到 iteration 2 |
| 当前 tensor-tree 路径，恢复验证 iteration 2 | `TRAIN_ITERS=2 SAVE_INTERVAL=1 RACER_VERIFY_LOAD_AFTER_SAVE=1` | 1.97 s | 1.92 s | 完整 `load_checkpoint()` 为 0.353 s | 最终 checkpoint 加载验证通过 |

两次恢复验证中，RACER load 的 tensor rebuild 部分约为 0.309 s。运行结束后 `/workspace/checkpoints` 仍为 4 KB，没有写出 checkpoint 大文件。

## 当前默认值

`train_gpt2_1_5b.sh` 默认使用：

- `CUDA_VISIBLE_DEVICES=1,2,3,4,5`
- `NPROC_PER_NODE=4`
- `RACER_ENABLE=1`
- `RACER_DISTRIBUTED_STORE=1`
- `RACER_ASYNC_OFFLOAD=0`

## 100 次迭代 / 10 次 checkpoint 验证

命令形态：通过 `/workspace/Megatron-LM-FT/train_gpt2_1_5b.sh` 运行 `TRAIN_ITERS=100 SAVE_INTERVAL=10 RACER_VERIFY_LOAD_AFTER_SAVE=1 RACER_RETAIN_CHECKPOINTS=1`。完整日志：`/workspace/logs/gpt2_1.5b_racer_100x10_retain1/run.log`。详细报告：`/workspace/Megatron-LM-FT/racer_checkpoint_100x10_report.md`。

| 指标 | 次数 | 平均值 | 中位数 | 最小值 | 最大值 |
| --- | ---: | ---: | ---: | ---: | ---: |
| RACER store 总时间 | 10 | 1.708 s | 1.437 s | 1.389 s | 2.947 s |
| Store 元数据 | 10 | 3.48 ms | 3.49 ms | 3.34 ms | 3.63 ms |
| Store tensor view 准备 | 10 | 11.73 ms | 11.68 ms | 11.54 ms | 12.20 ms |
| Store RACER 调用/NCCL 路径 | 10 | 1.695 s | 1.425 s | 1.377 s | 2.935 s |
| 阻塞 Megatron `save_checkpoint()` 的时间 | 10 | 1.844 s | 1.588 s | 1.471 s | 3.105 s |
| 阻塞训练循环的 `save-checkpoint` timer | 10 | 1.851 s | 1.593 s | 1.485 s | 3.119 s |
| RACER load 适配层总时间 | 10 | 279.40 ms | 277.55 ms | 266.90 ms | 307.86 ms |
| Load RACER 拉取 | 10 | 265.65 ms | 263.88 ms | 253.27 ms | 294.33 ms |
| Load tensor 物化 | 10 | 15.61 ms | 14.99 ms | 14.57 ms | 18.36 ms |
| Load tree 解码 | 10 | 0.98 ms | 0.96 ms | 0.77 ms | 1.23 ms |
| 完整 Megatron `load_checkpoint()` 验证 | 10 | 307.25 ms | 303.96 ms | 290.80 ms | 349.74 ms |

所有预期 checkpoint iteration 都被观察到：10、20、30、40、50、60、70、80、90、100。每次保存后都成功执行了 Megatron `load_checkpoint()` 验证。checkpoint 目录 `/workspace/checkpoints/gpt2_1.5b_racer_100x10_retain1` 保持 4 KB；RACER 在内存中只保留 1 个 checkpoint，并在运行过程中清理了 9 个旧 checkpoint。
