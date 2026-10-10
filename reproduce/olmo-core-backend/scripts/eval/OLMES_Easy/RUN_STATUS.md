# Dense单点SiLU：转换后HF权重GPU评测

2026-10-08 21:49（UTC+8）更新。五条agent评测均完整完成，每个任务1×8卡、priority1、xtuner-前缀。最终调度器确认Succeeded/active=0/succeeded=1，评测文件核验192/192任务，共960项。

| 模型 | RJob ID | 状态 | 实际耗时，分钟 | GPU平均利用率 |
| --- | --- | --- | ---: | ---: |
| dense | `xtuner-olmes-easy-dense-r2-1008-2132-7821394` | Succeeded；192/192 | 14.15 | 41.1% |
| q | `xtuner-olmes-easy-q-r2-1008-2132-9164594` | Succeeded；192/192 | 13.63 | 44.4% |
| k | `xtuner-olmes-easy-k-r2-1008-2132-10507094` | Succeeded；192/192 | 15.04 | 42.3% |
| up | `xtuner-olmes-easy-up-r2-1008-2132-11809105` | Succeeded；192/192 | 14.65 | 41.7% |
| gate | `xtuner-olmes-easy-gate-r2-1008-2132-13070245` | Succeeded；192/192 | 14.29 | 44.3% |

全部使用BF16、DP8/TP1、每卡batch32；40个worker均第一次成功，无OOM降batch。各模型192项task hash及primary metric与Dense完全一致，分数均有限。日志未发现未加载/随机初始化权重、形状不匹配、OOM或NaN告警。

耗时包含模型初始化、任务准备和汇总，不含排队/容器启动。GPU利用率为15秒粗采样的全程均值，含初始化和空闲，仍有提升空间。

[完整分数、速度及GPU报告](results/dense_single_silu_20261008/GPU_RESULTS.md)／[192项逐任务对照CSV](results/dense_single_silu_20261008/task_scores.csv)／[最终调度器状态](final_gpu_status_20261008.json)／[完整任务及attempt记录](dense_single_silu_eval_jobs.json)。

首轮五个任务在推理前因OLMES拒绝顶层use_cache参数而失败，全部确认Failed/active=0后重提r2。r2使用独立评分config关闭cache，原HF config和权重保持不变，首轮输出保留；评测关闭W&B，没有待同步段。

本次核实转换后的HF模型能完整运行OLMES；原生OLMo/HF数值forward对照尚未运行，不能把benchmark完成视为已验证数值等价。五条评测均结束，本批不再需要轮询；此前训练任务不受影响。
