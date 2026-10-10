# Dense / C0 / C6：实际训练步效率微基准

2026-10-07 04:22 HKT。任务 `mose-eff-step-r1-p1-1x8-pool-1007-0406-44a9-70355` 确认Succeeded、active0/succeeded1，三组结果各自经SHA256校验。三次独立8-rank进程顺序运行在同一个H200节点，均观察到有限训练CE及梯度。

| 配置 | 参数量 | 平均完整训练步(s) | 全局tokens/s | 相对Dense吞吐 | PyTorch峰值allocated(GiB) | 峰值reserved(GiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dense | 1,484,916,736 | 3.690175 | 568,307 | 1.000× | 51.269 | 57.861 |
| c0 | 797,050,880 | 2.901662 | 722,742 | 1.272× | 56.488 | 62.957 |
| c6 | 797,050,880 | 2.987917 | 701,878 | 1.235× | 66.305 | 72.879 |

**C6在本次合成输入训练步中吞吐约为Dense的1.235×，但峰值allocated显存约为1.293×。** 相对C0，C6吞吐约低2.9%，峰值allocated约高17.4%。不能把参数减少46.3%写成实际显存减少或同比加速；显存增加的具体来源尚未做profiler归因。

## 测量口径

- 同一节点8张H200、全局batch256、seq8192、每卡microbatch32768 tokens；每卡32条序列拆为八次microbatch累积，然后真实梯度裁剪、Muon更新与FP8后处理。Dense旧训练microbatch65536，本次显式统一32768。
- 5个完整optimizer step预热后测10步；每步包含train_batch的forward/backward和optim_step，取最慢rank耗时。起始barrier和GPU同步对齐计时，末尾同步纳入时间。
- 输入为预先生成的固定随机token，各模型每个rank输入一致；不包含真实数据加载、eval、checkpoint或完整预训练的系统开销，不把随机数据CE纳入质量比较。
- 从头初始化模型与optimizer，仅job-local临时缓存；没有加载/保存任何源checkpoint，也没有写W&B。保留原30B调度horizon，测试的短循环不改变原训练任务。
- PyTorch峰值allocated/reserved在预热后重置，跨rank取最大值；不等同于NVML整卡总占用，也不能单独分解为参数、激活、编译缓存或通信开销。
- 单个节点、固定运行顺序和10个相关测量步不足以证明跨硬件稳定加速；本次结果是效率线索，不是端到端训练速度的结论。

## 原始步耗时与来源

| 配置 | 最小(s) | 中位(s) | 最大(s) | 样本标准差(s) |
| --- | ---: | ---: | ---: | ---: |
| dense | 3.683689 | 3.689688 | 3.698855 | 0.004657 |
| c0 | 2.892930 | 2.902231 | 2.907335 | 0.005320 |
| c6 | 2.980713 | 2.988147 | 2.995964 | 0.005744 |

软件：PyTorch 2.10.0+cu128，CUDA 12.8。训练步计数、参数、全局batch、microbatch、硬件/软件、有限CE/梯度及吞吐算术由`collect.py`复核。

数据：[汇总JSON](SUMMARY.json)、[Dense](DENSE_STEP_EFFICIENCY.json)、[C0](C0_STEP_EFFICIENCY.json)、[C6](C6_STEP_EFFICIENCY.json)。完成状态和校验值：[jobs.json](jobs.json)。完成日志：`completion_observation.log`。
