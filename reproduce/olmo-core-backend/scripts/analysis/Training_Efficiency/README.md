# 实际训练步效率：Dense / C0 / C6

在同一个pool8节点上依次运行三个独立分布式进程，使用原模型、Muon、FP8、编译与FSDP设置。全局batch256、seq8192，每卡microbatch32768 tokens，八次梯度累积后执行一次optimizer step。Dense旧训练microbatch65536，本次显式统一32768。

每个模型从头初始化，只使用job-local临时工作目录。没有加载或保存checkpoint，不运行完整预训练，不写W&B。保留30B调度horizon；5个完整optimizer step预热后测10步。预先生成的每卡32条固定随机token序列在各模型间一致；train_batch实际生成标签并完成forward/backward，随后执行真实梯度裁剪、Muon更新及FP8后处理。

报告各步最慢rank的耗时、全局tokens/s、每GPU tokens/s、PyTorch峰值allocated/reserved显存。GPU同步和起始barrier用于对齐计时；编译预热、数据加载、评估和checkpoint排除，因此是合成输入的训练步微基准，不能冒充完整训练吞吐，也不将随机数据loss加入质量排名。

所有模型使用同一个8GPU节点，各案例独立缓存和进程。结果需由实际GPU执行、有限CE/梯度、JSON/SHA256和Succeeded/active0证据确认。单次节点/固定案例顺序存在噪声，不能据此宣称跨硬件稳定加速。
