# F3qk 与已完成 P2 配对、历史 C0/C2/C6 的完整 Train CE 比较

2026-10-06 23:06 HKT。F3qk 已完成14306步/30B，scheduler Succeeded、active0、succeeded4；最终 checkpoint 于22:34:32保存，22:34:35 Training complete。只读分析任务也已 Succeeded/active0；完整 JSON 从分段日志重建，SHA256与任务输出一致。

所有模型797,050,880参数。六个历史都完整覆盖steps1–14306。共同异常掩码只有step12516，来自历史C6的零CE记录；全程均值采用14305个共同点。末1000步为13307–14306，没有异常，全部采用1000个点。相比旧C0–C7报告，此次没有排除其他配置C3的step14304，因此历史C6从2.474827变为2.474763；不能混用两个掩码的均值做微小差异比较。

| 配置 | 全程平均 Train CE | 末1000步平均 Train CE | 最后14306步原始 CE | Validation macro CE@14000 | 末期差值 vs C6 |
|---|---:|---:|---:|---:|---:|
| C0（历史） | 2.793313975 | 2.478978409 | 2.508265495 | 2.740453558 | +0.004215442 |
| C2（历史） | 2.791298714 | 2.475667983 | 2.505400181 | **2.734791561** | +0.000905017 |
| **C6（历史）** | **2.789407092** | **2.474762967** | **2.504258633** | 2.736803120 | 0 |
| F3qk | 2.791282485 | 2.477421343 | 2.507214069 | 2.737547289 | +0.002658376 |
| P2qk | 2.795380606 | 2.481272976 | 2.510764122 | 2.741835692 | +0.006510010 |
| P2 | 2.798085070 | 2.484496541 | 2.513828516 | 2.746519273 | +0.009733575 |

**目前 C6 仍是已分析配置中末期 Train CE 最好的低秩候选；F3qk没有超过C6或C2。** F3qk比C0好0.001557，但它同时改变Q激活、UG共享和秩分配，不能将差值单独归因于Down变宽。F3qk与C2的全程均值几乎相同，末期则差0.001753；按预定末1000步主指标，C2更好。验证集仍以C2最好。

F3qk：Attention各rank512、不共享；Q/K SiLU、V/O Identity；UG共享480、中间Identity，原生SwiGLU保留；Down672、无gain RMSNorm。与它配对的F3仅Q/K为Identity，仍在训练，Q/K收益需等F3完成后比较。F4qk（共享UG400/Down816）和F5qk（共享UG560/Down528）补同一共享范围内的预算分配曲线，不把尚未完成的结果计入排名。

这是单种子质量筛选。P2曾从32卡续训至16卡，rank RNG未恢复；P2qk/F3qk连续32卡。历史C0/C2/C6使用先前保存的完成记录和本次重新读取的完整W&B历史，未声称今天重新验证了它们的scheduler状态。验证集为相同step14000、11源等权macro；旧evaluation口径限制仍保留。

数据：[F3QK_WITH_REFERENCES_FULL_TRAIN_CE.json](F3QK_WITH_REFERENCES_FULL_TRAIN_CE.json)。完成与日志证据：`../FFN_Rank_Reallocation/completion_f3qk.log`、`reports/f3qk_reference_completion.log`、`historical_reference_segments.json`。历史引用不是新的训练attempt，也没有用于宣称三种子统计。

W&B：[F3qk](https://wandb.ai/archspace/MoSE/runs/olmo3-1b-ffn-budget-f3qk-ug480-down672-shared-qk-silu-ug-linear-down-rms-fp8-30b)。
