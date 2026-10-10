# 当前实验与监控清单

状态快照：2026-10-10T22:02:18.439144+08:00（北京时间）。本清单来自实际RJob状态与日志。

当前六条训练C9、L1、R0–R3均已完成30B/14306步；六条均Succeeded、active=0，最终step14306保存成功与Training complete已核实。没有本轮仍在训练或排队的任务。完整末期Train CE尚待汇总；C9后续HF转换与OLMES仍待处理。

| 实验 | 配置与目的 | 最新状态 | 最新日志进度 |
| --- | --- | --- | --- |
| C9 | 独立rank512；Q-only SiLU，Down RMS，其余中间Linear | 已完成 | step14306/14306，最终保存已核实；最近上报step14300 CE 2.504 |
| L1 | 原FFN宽度8192、rank512；Up→Down latent残差 | 已完成 | step14306/14306，最终保存已核实；最近上报step14300 CE 2.505 |
| R0 | FFN宽度2048、rank1280；无新增残差 | 已完成 | step14306/14306，最终保存已核实；最近上报step14300 CE 2.555 |
| R1 | 同R0；新增两条2048宽残差 | 已完成 | step14306/14306，最终保存已核实；最近上报step14300 CE 2.63 |
| R2 | 同R0；新增Up→Down latent残差 | 已完成 | step14306/14306，最终保存已核实；最近上报step14300 CE 2.555 |
| R3 | 同R0；宽残差与latent残差均开启 | 已完成 | step14306/14306，最终保存已核实；最近上报step14300 CE 2.631 |

六条模型均为797,050,880参数，UG与Attention不共享。C9仅Q的低秩中间为SiLU；L1/R0–R3的Attention与UG中间全部Linear；所有Down为无gain RMSNorm，原生SwiGLU门控保留。

## 固定训练配方

OLMo3原骨架，16层、残差维度2048；seq8192、global batch256（2,097,152 tokens），每卡microbatch32768、8卡梯度累积8次；30B／14306步，Muon LR0.005，WSD warmup2000／末20% decay，FP8。model seed0、运行seed12536、数据seed34521；checkpoint每1788步、保留全部。

## 调度任务和W&B配置名称

以下W&B名称为训练配置中的完整名称；排队任务尚未初始化W&B，不表示已上传。调度job ID与W&B名称是不同标识。

| 实验 | RJob ID | W&B完整实验名称 |
| --- | --- | --- |
| C9 | `xtuner-mose-c9-q-rms-p1-1x8-1009-r1-mv0mlyc4-1b571` | `olmo3-1b-r512matched-c9-q-silu-down-rms-fp8-30b` |
| L1 | `mose-ffnres-l1-p1-1x8-1009-1601-8558-1220497` | `olmo3-1b-ffn-residual-l1-m8192-r512-unshared-linear-down-rms-fp8-30b` |
| R0 | `mose-ffnres-r0-p1-1x8-1009-1601-2d86-37284085` | `olmo3-1b-ffn-residual-r0-m2048-r1280-unshared-linear-down-rms-fp8-30b` |
| R1 | `mose-ffnres-r1-p1-1x8-1009-1601-1eb9-88370018` | `olmo3-1b-ffn-residual-r1-m2048-r1280-unshared-linear-down-rms-fp8-30b` |
| R2 | `mose-ffnres-r2-p1-1x8-1009-1601-3788-1295629` | `olmo3-1b-ffn-residual-r2-m2048-r1280-unshared-linear-down-rms-fp8-30b` |
| R3 | `mose-ffnres-r3-p1-1x8-1009-1601-c6f0-34872256` | `olmo3-1b-ffn-residual-r3-m2048-r1280-unshared-linear-down-rms-fp8-30b` |

## 监控与后续工作

- 监控间隔1800秒，包含以上6项。排队或SSH超时不重复提交；后台监控的实际状态记录在下列JSON及日志。
- 启动后核实8 ranks／dp_replicate=1、dp_shard=8、step0保存、实际W&B offline段、首步与有限CE。R0–R3另比较实际初始化SHA256。
- 终态需核实active=0、step14306保存成功与Training complete；停止段先同步精确W&B记录，再处理授权重启。
- 新残差架构已补统一HF adapter和独立runtime。meta参数/结构审计及8项原生行为检查通过；HF数值forward验证仍为not_run。
- C9已完成训练，待转换最终checkpoint并补同套OLMES 192项benchmark；L1/R0–R3先比较完整末1000步Train CE，不因启动或单步CE提前判定优劣。

[L1与R0–R3详细公式、预算、代码和入口](scripts/train/FFN_Residual_Bottlenecks/README.md)／[C9完整配置](scripts/train/Rank512Matched_Combinations/C9_Q_ONLY_RMS.md)。

[实时状态JSON](scripts/train/FFN_Residual_Bottlenecks/latest_status.json)／[任务台账](scripts/train/FFN_Residual_Bottlenecks/jobs.json)／[运行与监控历史](scripts/train/FFN_Residual_Bottlenecks/RUN_STATUS.md)／[C9状态](scripts/train/Rank512Matched_Combinations/C9_RUN_STATE.json)。

已取消新增C10：所需K-only SiLU＋Down RMS、独立rank512配置就是已完成的C0，后续直接复用C0结果。
