# Rank512激活与C0–C8：OLMES评测原型和批次

使用原来的 `olmo3:base_easy` 完整192项，不补重复种子。14个原始训练记录均 finished，最终step14306，30B tokens。以下结构来自归档的训练配置，转换以实际checkpoint保存配置及DCP元数据为准。所有14个模型均为797,050,880参数。

第一组5个模型每个投影独立rank512；C0–C8保持相同总参数预算，共享后rank随配置调整。这里的Linear/SiLU/RMS只描述低秩两因子之间的变换；原生SwiGLU的外层SiLU和OLMo原有归一化仍保留。

| 标签 | 原始训练配置 | W&B完整实验名称 |
| --- | --- | --- |
| COLA | CoLA / 全部SiLU | [olmo3-1b-down-activation-silu-fp8-30b](https://wandb.ai/archspace/MoSE/runs/olmo3-1b-down-activation-silu-fp8-30b) |
| FULL_LINEAR | Full-Linear | [olmo3-1b-all-lowrank-linear-fp8-30b](https://wandb.ai/archspace/MoSE/runs/olmo3-1b-all-lowrank-linear-fp8-30b) |
| RMS_DOWN | RMS@Down，其余SiLU | [olmo3-1b-down-activation-rms-fp8-30b](https://wandb.ai/archspace/MoSE/runs/olmo3-1b-down-activation-rms-fp8-30b) |
| Q_ONLY | Q-only SiLU，Down Linear | [olmo3-1b-all-linear-q-only-silu-fp8-30b](https://wandb.ai/archspace/MoSE/runs/olmo3-1b-all-linear-q-only-silu-fp8-30b) |
| K_ONLY | K-only SiLU，Down Linear | [olmo3-1b-all-linear-k-only-silu-fp8-30b](https://wandb.ai/archspace/MoSE/runs/olmo3-1b-all-linear-k-only-silu-fp8-30b) |
| C0 | K SiLU；不共享；Down RMSNorm | [olmo3-1b-r512matched-c0-k-silu-down-rms-fp8-30b](https://wandb.ai/archspace/MoSE/runs/olmo3-1b-r512matched-c0-k-silu-down-rms-fp8-30b) |
| C1 | K SiLU；KV共享；Down RMSNorm | [olmo3-1b-r512matched-c1-k-silu-down-rms-kv-fp8-30b](https://wandb.ai/archspace/MoSE/runs/olmo3-1b-r512matched-c1-k-silu-down-rms-kv-fp8-30b) |
| C2 | K SiLU；UG共享；Down RMSNorm | [olmo3-1b-r512matched-c2-k-silu-down-rms-gateup-fp8-30b](https://wandb.ai/archspace/MoSE/runs/olmo3-1b-r512matched-c2-k-silu-down-rms-gateup-fp8-30b) |
| C3 | K SiLU；KV+UG共享；Down RMSNorm | [olmo3-1b-r512matched-c3-k-silu-down-rms-kv-gateup-fp8-30b](https://wandb.ai/archspace/MoSE/runs/olmo3-1b-r512matched-c3-k-silu-down-rms-kv-gateup-fp8-30b) |
| C4 | Q/K/V/O SiLU；KV共享；Down RMSNorm | [olmo3-1b-r512matched-c4-attention-silu-down-rms-kv-fp8-30b](https://wandb.ai/archspace/MoSE/runs/olmo3-1b-r512matched-c4-attention-silu-down-rms-kv-fp8-30b) |
| C5 | Q SiLU；UG共享；Down RMSNorm | [olmo3-1b-r512matched-c5-q-silu-down-rms-gateup-fp8-30b](https://wandb.ai/archspace/MoSE/runs/olmo3-1b-r512matched-c5-q-silu-down-rms-gateup-fp8-30b) |
| C6 | Q/K SiLU；UG共享；Down RMSNorm | [olmo3-1b-r512matched-c6-qk-silu-down-rms-gateup-fp8-30b](https://wandb.ai/archspace/MoSE/runs/olmo3-1b-r512matched-c6-qk-silu-down-rms-gateup-fp8-30b) |
| C7 | Q/K及Up/Gate SiLU；UG共享；Down RMSNorm | [olmo3-1b-r512matched-c7-qk-silu-ug-silu-down-rms-gateup-fp8-30b](https://wandb.ai/archspace/MoSE/runs/olmo3-1b-r512matched-c7-qk-silu-ug-silu-down-rms-gateup-fp8-30b) |
| C8 | Q/K/V/O及Up/Gate Linear；UG共享；Down RMSNorm | [olmo3-1b-r512matched-c8-linear-down-rms-gateup-fp8-30b](https://wandb.ai/archspace/MoSE/runs/olmo3-1b-r512matched-c8-linear-down-rms-gateup-fp8-30b) |

Q-only/K-only的Down为Identity，不能用C0（Down RMS）替代K-only。Full-Linear不共享且Down Identity，不能用C8（UG共享、Down RMS）替代。CoLA和RMS@Down使用Down激活四格中的S/R完整30B版本；之前已评测的Dense单点Q/K（rank1024、约1.485B参数）属于另一组，不能替代这里的rank512模型。

| C配置 | Q/K/V/O rank | UG rank及共享 | Down rank |
| --- | --- | --- | --- |
| C0 | 512,512,512,512 | 512，独立 | 512 |
| C1 | 512,688,688,504 | 512，独立 | 512 |
| C2 | 512,512,512,496 | 568，共享 | 520 |
| C3 | 520,680,680,512 | 568，共享 | 512 |
| C4 | 512,688,688,504 | 512，独立 | 512 |
| C5 | 512,512,512,496 | 568，共享 | 520 |
| C6 | 512,512,512,496 | 568，共享 | 520 |
| C7 | 512,512,512,496 | 568，共享 | 520 |
| C8 | 512,512,512,496 | 568，共享 | 520 |

原型来源和checkpoint候选路径见 [cohort manifest](../../convert/rank512_c0_c8_cohort.json)。前5个的路径直接来自归档trainer.save_folder；C组使用既有标准保存路径，必须在调度任务内核实实际保存配置、step、DCP键名/形状和唯一参数量，失败则明确记录，不回退旧step或替换模型。14/14实际checkpoint已在转换任务预检查中核实：step14306、保存配置、DCP模型元数据、原生/HF参数形状和797,050,880唯一参数量全部通过；见[预检查记录](../../convert/rank512_c0_c8_preflight.json)。数值验证未运行。

统一HF转换使用 `export_manifest_cohort.py` 调用共享 `hf_export.export_checkpoint`，没有每个模型自己的转换实现。dtype auto保留master权重，OLMES统一加载BF16。原生/HF数值对齐验证沿用先前要求，状态为not_run。

每个评测为agent分区1×8卡、低优1、xtuner-前缀，DP8/TP1、每GPU batch32，保留原任务/样本/few-shot设置。OOM仅对失败worker缩batch，已完成任务不重算。任务账本：[rank512_c0_c8_eval_jobs.json](rank512_c0_c8_eval_jobs.json)。每30分钟汇总监控；只有192任务完整和调度器终态同时核实后才记完成。

准确率与BPB分别汇总，不合成一个混合分数。之前已完成的Dense可以作为较大参数预算的参考，不纳入“同参数”对照。

转换任务已于2026-10-09 03:34:36+08:00提交：`xtuner-mose-hf-r512-c0-c8-1009-r1-76211050`，agent 1×8卡、低优1；03:35:19调度器为Inqueue，active=0，尚未核实转换完成，评测任务尚未提交。

用户于2026-10-09要求先不运行、等说好统一启动。已于03:36:57停止转换任务；停止时COLA已导出、Full-Linear正在转换、其余未开始转换，评测14项均未提交。已有产物保留，后续启动须核实停止状态并使用新输出目录，不覆盖或把部分产物视为完整批次。

03:37:34调度器已确认Stopped，active=0、stopped=1。后续不再轮询或提交，等待用户明确启动。

2026-10-09 03:41用户重新授权启动。转换r2任务：`xtuner-mose-hf-r512-c0-c8-1009-r2-5639787`，agent1×8、priority1，使用新HF输出根目录，保留停止批次。14模型均用同一套192项OLMES；完成后继续半小时监控DU/DG/DUG三条arch pool训练。

恢复批次：14/14 HF转换已完成，转换任务Succeeded/active=0。14个OLMES任务均已提交；首次快照5条Running、9条Inqueue。完整导出记录见[14模型r2导出记录](../../convert/rank512_c0_c8_exports_r2.json)，最新状态以任务账本为准。

2026-10-09 04:28已确认14份OLMES均完成192项，全部Succeeded/active=0。[完整分数和解读](results/rank512_c0_c8_20261009/METRIC_SUMMARY.md)。接下来只监控DU/DG/DUG三条既有训练，每30分钟核查直到完成。
