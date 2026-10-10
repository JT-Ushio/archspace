# OLMES Base Easy：8卡并行与耗时均衡

此入口使用已有独立 OLMES 环境，对本地 HF 模型评测完整 `olmo3:base_easy`：QA RC、QA BPB、数学 BPB、代码 BPB。当前官方配置展开为192个子任务。支持单文件或分片 safetensors，适用于最近统一导出的 Dense 与单点 SiLU 权重。

## 为什么旧脚本慢

旧 SiTU-Up 脚本已经是 DP8 / TP1、每卡 batch32，不需要再从 TP8 改一次。旧驱动按任务编号取模分配，24个任务分给一张卡，却没有考虑任务耗时。

从2026-08-31的8份日志恢复的 worker 耗时如下。这是实际日志时间间隔，未包括容器排队时间。

| Worker | 旧实际耗时，秒 | 分钟 |
| --- | ---: | ---: |
| 0 | 344 | 5.7 |
| 1 | 691 | 11.5 |
| 2 | 2269 | 37.8 |
| 3 | 553 | 9.2 |
| 4 | 759 | 12.7 |
| 5 | 805 | 13.4 |
| 6 | 971 | 16.2 |
| 7 | 440 | 7.3 |

SQuAD 的 RC/BPB 各约620秒，DROP 各约500–558秒，HellaSwag 各约276秒。它们集中在少数 worker 上，造成其他卡先结束而整场评测仍在等待。

注意 OLMES `metrics.json` 的 `processing_time` 在这版代码中不包含主要模型推理，不能直接据此分配 GPU 工作。本目录 [historical_timings.json](historical_timings.json) 使用 `Processing task` 到下一任务/最终汇总的日志墙钟间隔，共192项，包含推理和任务准备。

## 本次优化

1. 保持 DP8 / TP1，每张卡加载一份完整模型。启动 Python 之前设置单卡 `CUDA_VISIBLE_DEVICES`，保留原分配的非连续卡号或 GPU UUID，避免8个进程集中到 GPU0。
2. 默认按历史耗时从长到短排列，每次把任务交给当前预计负载最低的 worker。保留每个任务的原始全局 ID，汇总结果时恢复顺序并核验完整性。`--assignment round-robin` 可重建轮询分配作为速度对照。
3. 默认每卡 batch32，保留历史已跑通设置。可设16/64；出现明确 CUDA OOM 时，只把失败 worker 的 batch 减半重启，沿用同一模型和 task spec，保留其已完成任务。不对其他错误重试。最终逐 worker 记录实际 batch 和每次尝试。
4. 在输出目录创建独立的 `scoring_model/config.json`，对这些打分任务设置 `use_cache=false`，避免保存本次计算用不到的生成 KV cache。其余权重、tokenizer及runtime文件链接回原HF目录，源文件保持不变。此版OLMES模型参数白名单不接受顶层`use_cache`，因此通过独立config设置。模型dtype仍为 BF16，原 task 的 split、few-shot、limit、context 和指标定义保持原配置。
5. tmp 与 HF 动态模块缓存使用节点本地 `/tmp`，各 worker 独立模块缓存。HF 数据缓存继续使用已有共享离线缓存。关闭原始请求的大量 JSONL 输出，保留结果与预测。
6. 输出总耗时、每个 worker 耗时、实际 batch、任务耗时，以及每15秒的 `nvidia-smi` GPU利用率/显存/功率采样。GPU采样是粗粒度利用率，不能等同于 Tensor Activity 或 SM Active 的精确采样。

已有192任务历史耗时全部匹配；负载均衡方案预计每卡约839–841秒，即14分钟左右。这是旧模型历史耗时的排程估算，不保证其他模型都在20分钟内完成。2026-10-08五个新导出模型的GPU实测已完成，每个完整192任务耗时13.63–15.04分钟，含模型初始化、任务准备及汇总，不含排队和容器启动。全部40个worker均保持batch32，无OOM重试。GPU全程采样均值41.1–44.4%，包含初始化与任务准备的空闲，不能表述为全程利用率拉满。[完整GPU结果](results/dense_single_silu_20261008/GPU_RESULTS.md)。BF16的batch或缓存计算路径调整仍可能产生小量浮点差异，本轮未与旧驱动做同模型分数逐项对照。

OLMES会为任务/样本建立独立的 few-shot 和 subsampling RNG，驱动沿用这些默认值，不通过改变seed、截短上下文、减少题数或省略任务提速。`PYTHONHASHSEED`默认0，使suite展开顺序可重复；manifest仍记录实际解析出来的完整任务配置与映射。

## 使用方法

推荐统一入口：[run_easy.sh](run_easy.sh)。使用同一入口传入任意支持的本地 HF 模型，不为每个模型建立一份评测实现。

```bash
OLMES_BATCH_SIZE=32 bash scripts/eval/OLMES_Easy/run_easy.sh \
  /mnt/shared-storage-gpfs2/intern-pretrain-shared02/shijiayang-p/ckpts/huggingface/dense-single-silu-step14306-20261008/dense-step14306
```

路径为GPU容器中的GPFS2挂载路径；本地环境自动选择 `shared-storage-user` 别名。也可用 `OLMES_GPFS_ROOT`、`OLMES_ENV`、`OLMES_REPO`、`OLMES_CACHE_ROOT` 覆盖环境/缓存位置。

设置 `OLMES_BATCH_SIZE=64` 可尝试更大 batch，默认启用 OOM 降 batch。正式比较32/64时使用同一个模型、相同环境与任务集、两个全新输出目录，比较总 wall time、worker尾部耗时、GPU占用和统一按alias对齐的分数，不能只看某个瞬时GPU利用率。

脚本默认创建带模型名、时间和PID的新输出目录，也可指定 `OLMES_OUTPUT_DIR`。非空输出目录会拒绝，避免混入旧评测结果。OOM内部重试使用已生成的同一spec，不属于重新规划旧目录。

只生成配置和分片、不加载模型权重、不运行forward：

```bash
bash scripts/eval/OLMES_Easy/run_easy.sh /absolute/HF/model/directory --plan-only
```

旧 [SiTU-Up入口](../../../legacy/run_OLMES_Easy_SiTU_Up_Step75080_DP8_BS32.sh) 已改为调用统一入口，仍默认选择旧step75080模型，可通过 `OLMES_MODEL_DIR` 改模型；默认输出使用新目录，原来的完整评测结果保留。原 `tools/olmes_runtime/run_easy_dp_sharded.py` 也作为兼容入口跳转到本目录实现。维护代码放在 `scripts/eval/`，避免旧 `tools/` 整体被 Git 忽略造成实现遗漏。

## 输出及后续耗时校准

| 文件 | 内容 |
| --- | --- |
| `run_config.json` | 原模型/评分模型路径、OLMES路径、套件、精度、并行方式、batch配置 |
| `scoring_model/` | 独立评分config；其余文件链接到原HF产物 |
| `shard_manifest.json` / `task_specs/*.jsonl` | 原始task配置、alias、全局/局部映射、预计worker耗时 |
| `logs/shard-*.log` | 每个GPU worker日志，OOM重试追加在同一个文件 |
| `shards/shard-*` | OLMES原生逐任务指标和预测 |
| `metrics.json` / `metrics-all.jsonl` | 按全局ID恢复顺序后的完整结果 |
| `evaluation_complete.json` | 全部任务、唯一ID和alias核验通过后写入 |
| `execution_report.json` | 实际wall time、各worker完成耗时、batch与尝试记录 |
| `gpu_samples.jsonl` | index/UUID/GPU utilization/memory used/memory total/power，每15秒 |
| `task_timings.json` | 新一轮日志墙钟耗时；有重复重试marker的worker不推测任务映射 |

后续可传 `--timings /path/to/task_timings.json`，以同一架构的实测耗时重新均衡。没有历史耗时的新增任务使用已知任务耗时中位数估计，同时记录覆盖率；不把估计当作实测。

[collect_gpu_results.py](collect_gpu_results.py)在所有任务完成后按准确alias汇总多个模型，检查完整任务集、有限分数、相同primary metric及模型路径。套件分数按官方suite树递归宏平均，避免把MMLU的57个子集直接作为57份QA权重，也不混合accuracy与BPB。生成逐任务CSV、运行/GPU JSON及Markdown结果：

```bash
/path/to/conda_envs/olmes/bin/python scripts/eval/OLMES_Easy/collect_gpu_results.py \
  --olmes-repo /path/to/archspace/olmes_official \
  --ledger scripts/eval/OLMES_Easy/dense_single_silu_eval_jobs.json \
  --output-dir scripts/eval/OLMES_Easy/results/dense_single_silu_20261008
```

## 本次验证与提交状态

- CPU调度/结果合并/实际子进程模拟OOM重试/源HF产物不可变检查：8个测试通过。
- 使用现有 OLMES 环境和最近 Dense HF目录执行 `--plan-only`：192个任务全部解析成功，alias集合与旧完整结果一致，历史耗时覆盖192/192。
- Python编译、Ruff检查与Bash语法检查通过。
- 2026-10-08用户明确授权在agent分区GPU实测；Dense与Q/K/Up/Gate单点SiLU五条1×8卡评测均完整完成192项，均为priority1、`xtuner-`前缀。[任务及状态](RUN_STATUS.md)／[完整ledger](dense_single_silu_eval_jobs.json)／[实测分数与速度](results/dense_single_silu_20261008/GPU_RESULTS.md)。
- 首轮任务在推理前因顶层`use_cache`不在OLMES白名单内失败；五条均确认terminal/active=0后，改用独立评分config并重提r2，不重复已有活跃任务。
- 尚未运行原生OLMo/HF的数值forward对照；benchmark推理成功不等同于数值一致性验证。

本批评测遵从用户最新授权：**agent分区，每个任务1×8卡，priority1，任务名前缀 `xtuner-`，30分钟监控一次。** 训练任务仍遵从此前的arch pool约束，不在agent启动训练。
