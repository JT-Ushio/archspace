# MoSE checkpoint：只保留最终 step14306

用户于2026-10-10明确授权：所有实验仅保留最终`step14306`，清理完统计剩余容量；随后限定只处理MoSE文件夹，不删除其他目录。本规则取代同日较早方案中保留step0/step7152及静态引用step的建议。

## 本次唯一清理根目录

```text
/mnt/shared-storage-gpfs2/intern-pretrain-shared02/taoji/ckpts/Olmo3-MoSE
```

只处理该根目录下`<run>/stageN/step数字`中的非`step14306`目录。代码不接受其他根路径，不删除run/stage元数据，不处理任何兄弟目录、HF模型或个人历史CoLA/SiTU目录。

## 执行与例外

[清理脚本](scripts/analysis/Checkpoint_Prune_20261010/keep_final_only.py)默认只盘点；`--execute`才删除。执行前先生成包含具体路径、目录标识和容量的清单，执行时重新核对同一清单及最终点，再逐项记录删除结果。

无可读有效step14306、存在更晚step、软链接或扫描错误的实验单列为例外。初步布局核实不等于模型数值验证；本次不运行GPU forward或OLMES。

只清理完整step目录，最终点保留全部模型、优化器、trainer和配置文件。删除的中间点不再能用于恢复或中期权重分析；训练曲线与已有分析产物继续保留。

## 结果

2026-10-10 22:44:34 HKT完成清理，22:45:30 HKT完成事后核验。所有删除路径均位于上面的唯一根目录。

| 项目 | 实测结果 |
| --- | ---: |
| 清理前目录逻辑大小 | 7,944,356,195,298 bytes，约7.944 TB（7.225 TiB） |
| 删除中间checkpoint | 700个 |
| 目录逻辑大小减少 | 6,249,094,099,853 bytes，约6.249 TB（5.684 TiB） |
| 清理后目录逻辑大小 | 1,695,262,095,445 bytes，约1.695 TB（1.542 TiB） |
| 清理后目录allocated大小 | 1,697,759,727,616 bytes |
| 85个实验保留的最终点 | 各仅有step14306，合计756,108,991,195 bytes，约756.109 GB |
| 未处理的例外 | 26个实验，99个checkpoint，均无step14306；另有1个空stage |
| 剩余普通文件 | 53,313个；盘点无扫描错误、无软链接 |

已核实：700个目标目录全部删除；85个最终点存在且可读，相关配置/metadata的目录及文件标识保持一致，各自stage下只剩step14306；26个例外实验的99个checkpoint全部仍在。核验不包含GPU数值forward。

所在共享文件系统在22:45:30 HKT报告可用49,526,521,987,072 bytes（约49.53 TB）；这是整个挂载的可用量，不是个人配额。逻辑容量减少也不等同于底层快照/存储系统即时回收量。

证据：[执行清单](scripts/analysis/Checkpoint_Prune_20261010/final_only_execution/plan.json)、[逐项删除记录](scripts/analysis/Checkpoint_Prune_20261010/final_only_execution/deletion_events.jsonl)、[清理前后统计](scripts/analysis/Checkpoint_Prune_20261010/final_only_execution/result.json)、[事后核验](scripts/analysis/Checkpoint_Prune_20261010/final_only_execution/verification.json)。

## 尚待确定保留策略的26个实验

用户已被询问这些实验是保留各自最后一步、全部删除还是暂时保留。在得到回答前，本轮未删除它们。它们计入上面的1.695 TB剩余容量。

| 实验目录 | 剩余checkpoint数 | 当前最大step |
| --- | ---: | ---: |
| `olmo3-1b-MoSE-nolinear-rmsdown-weight-fp8` | 1 | 75080 |
| `olmo3-1b-SiTU-rms-up-c4-5e-3` | 2 | 27500 |
| `olmo3-1b-SiTU-up-5e-3` | 2 | 75080 |
| `olmo3-1b-baseline-5e-3` | 1 | 75080 |
| `olmo3-1b-baseline-fp8-5e-3` | 1 | 75080 |
| `olmo3-1b-cross-layer-incremental-memory-g2-r512-s16-fp8-30b` | 8 | 9500 |
| `olmo3-1b-cross-layer-incremental-memory-g2-r512-s32-fp8-30b` | 7 | 9000 |
| `olmo3-1b-cross-layer-kv-g2-r512-s32-down-rms-fp8-30b` | 1 | 0 |
| `olmo3-1b-cross-layer-shared-base-memory-g2-r512-s0-fp8-30b` | 7 | 9600 |
| `olmo3-1b-down-activation-rms-adamw-fp8-30b` | 8 | 12516 |
| `olmo3-1b-down-activation-silu-adamw-fp8-30b` | 8 | 12516 |
| `olmo3-1b-ffn-linear-attention-silu-fp8-30b` | 5 | 7152 |
| `olmo3-1b-kv-shared` | 3 | 10728 |
| `olmo3-1b-mla-morphnorm-ckv-layer-residual-r256-fp8-30b` | 7 | 9600 |
| `olmo3-1b-mose-cola-r1-512-r2-512-down-r1-256-down-r2-256-standard-swiglu-5e-3` | 2 | 27500 |
| `olmo3-1b-mose-cola-r1-512-r2-512-down-r1-256-down-r2-256-swiglu-5e-3` | 1 | 75080 |
| `olmo3-1b-per-layer-kv-latent-control-r512-fp8-30b` | 2 | 200 |
| `olmo3-1b-precision-down-rms-bf16` | 5 | 3577 |
| `olmo3-1b-precision-down-rms-fp8` | 4 | 3577 |
| `olmo3-1b-precision-down-silu-bf16` | 5 | 3577 |
| `olmo3-1b-precision-down-silu-fp8` | 4 | 3577 |
| `olmo3-1b-qk-shared` | 3 | 10728 |
| `olmo3-1b-qkv-shared` | 3 | 10728 |
| `olmo3-1b-ug-partial-p1-s560-p0-linear-down-rms-fp8-30b` | 1 | 0 |
| `olmo3-1b-ug-partial-p2-s480-p80-linear-down-rms-fp8-30b` | 5 | 7152 |
| `olmo3-1b-up-gate-shared` | 3 | 10728 |
