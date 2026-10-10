# Dense预算FFN Linear＋Down RMS：末期Train CE

2026-10-09从W&B读取四条任务的完整末1000步，严格使用step13307–14306。每条恰好1000个唯一、有限且正的Train CE值，不使用抽样、平滑或排除步骤。三条新增任务均完成30B并保存最终step14306 checkpoint。

| 配置 | 末1000步Train CE | 与Dense差值 | 最终step14306原始CE | 参数量 |
| --- | ---: | ---: | ---: | ---: |
| Dense | 2.357587371 | +0.000000000 | 2.388383150 | 1,484,916,736 |
| DU：Up Linear＋Down RMS | 2.378748731 | +0.021161360 | 2.407394648 | 1,482,819,584 |
| DG：Gate Linear＋Down RMS | 2.372440193 | +0.014852822 | 2.401164055 | 1,482,819,584 |
| DUG：Up/Gate独立Linear＋Down RMS | 2.380480801 | +0.022893430 | 2.408013821 | 1,481,771,008 |

三个组合的末期均值及最终原始CE均高于Dense，未带来训练CE改善。DG在三个新增组合中损失最小。原始最终CE只代表单步，结论主要依据同一末1000步均值。单种子比较，不作统计显著性声明。

DU只替换Up及Down：Up为两因子Linear低秩投影；Down为两因子之间加无gain RMSNorm的低秩投影，rank均为1632。Gate以及所有Q/K/V/O保留原生Dense结构，无共享。原生SwiGLU仍计算SiLU(gate)与up的逐元素乘积。DG和DUG按各自标签替换输入分支，其余沿用相同规则。三条都同时改变Down，没有同Dense预算的Down Linear配对对照，不能将质量差距单独归因于某个输入分支或Down归一化。

相对Dense的参数差异分别为−0.141230%、−0.141230%、−0.211845%，不是严格等参数。序列长度、全局batch、30B预算、Muon LR0.005、WSD warmup2000/末20%decay及FP8与Dense对照一致。新增任务microbatch32768 tokens，对照为65536；因子初始化遵循MoSE，未替换投影保留原生Dense初始化规则。没有声称step0全权重逐值相同。

本次API返回的Dense W&B状态字段为failed，历史仍包含完整13307–14306窗口；其末1000步均值及最终原始CE与此前[离线分段重建统计](../Dense_Single_SiLU/train_ce_comparison.json)精确一致。这个状态字段不用于代替训练步数和checkpoint完成证据。

| 配置 | W&B完整名称 |
| --- | --- |
| Dense | [olmo3-1b-dense-baseline-bs256-seq8k-30b-wsd-fp8-5e-3](https://wandb.ai/archspace/MoSE/runs/olmo3-1b-dense-baseline-bs256-seq8k-30b-wsd-fp8-5e-3) |
| DU | [olmo3-1b-dense-nearmatched-up-linear-r1632-down-rms-r1632-fp8-30b](https://wandb.ai/archspace/MoSE/runs/olmo3-1b-dense-nearmatched-up-linear-r1632-down-rms-r1632-fp8-30b) |
| DG | [olmo3-1b-dense-nearmatched-gate-linear-r1632-down-rms-r1632-fp8-30b](https://wandb.ai/archspace/MoSE/runs/olmo3-1b-dense-nearmatched-gate-linear-r1632-down-rms-r1632-fp8-30b) |
| DUG | [olmo3-1b-dense-nearmatched-up-gate-linear-r1632-down-rms-r1632-fp8-30b](https://wandb.ai/archspace/MoSE/runs/olmo3-1b-dense-nearmatched-up-gate-linear-r1632-down-rms-r1632-fp8-30b) |

[结构化统计](ce_results_20261009/TRAIN_CE_COMPARISON.json)；同目录保留四条完整末1000步原始数据。配置、公式和预算见[README](README.md)，训练完成及W&B同步证据见[RUN_STATUS](RUN_STATUS.md)。本次只读分析，无新增评测或训练任务。
