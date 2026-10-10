# Dense FFN Linear / Down RMS 任务状态

核验时间：2026-10-08 20:21 HKT。

用户最新约束：训练只提交到 arch pool，每个任务 **1 节点 × 8 GPU、priority=1**；不在 agent 分区启动训练。全局 batch256、seq8192、30B token 和学习率日程保持不变。

## 新提交的三个任务

| 实验 | 输入投影变更 | Down | 参数量 | 调度器状态 | 实际 Job ID |
| --- | --- | --- | ---: | --- | --- |
| DU | Up Linear 因子，Gate Dense | rank1632，无 gain RMS | 1,482,819,584 | Inqueue，active=0 | `xtuner-dense-ud-r1632-rms-1x8-p1-1008-2016-482868` |
| DG | Gate Linear 因子，Up Dense | rank1632，无 gain RMS | 1,482,819,584 | Inqueue，active=0 | `xtuner-dense-gd-r1632-rms-1x8-p1-1008-2017-3-25bd7` |
| DUG | Up/Gate 独立 Linear 因子，不共享 | rank1632，无 gain RMS | 1,481,771,008 | Inqueue，active=0 | `xtuner-dense-ugd-r1632-rms-1x8-p1-1008-2020-9e8f7` |

三条 submission 均已确认 created，随后逐条查询调度器确认 Inqueue；尚未启动，未产生 step0 或 W&B 训练段。没有在其他分区重复提交。rank1632 向下取整，参数量比 Dense 分别少0.1412% / 0.1412% / 0.2118%，属于 near-matched。原生 SwiGLU 的外层 SiLU(gate) 保留。完整配置、公式和 W&B run 名称见 [README](README.md)，提交元数据见 [jobs.json](jobs.json)。

## 当前需要监控的任务数

这两批已提交训练共8条；**7条尚未完成：4条运行中，3条排队中**。所有任务各申请8 GPU，运行中的4条共32 GPU；排队3条申请24 GPU。监控间隔沿用用户指定的1800秒。此处保存本次查询快照，不将其描述为已经启动新的自动汇报服务。

| 任务 | 状态 | active / succeeded | 实际 Job ID |
| --- | --- | --- | --- |
| W1 | Running | 1 / 0 | `mose-width-w1-p1-1x8-1008-0226-37ef-94920163` |
| W1-QK | Running | 1 / 0 | `mose-width-w1qk-p1-1x8-1008-0227-b83b-25485220` |
| W1-M | Running | 1 / 0 | `mose-width-w1m-p1-1x8-1008-0227-cf64-30995519` |
| G1 | Running | 1 / 0 | `mose-width-g1-p1-1x8-1008-0227-1760-42442674` |
| DU / DG / DUG | Inqueue | 均0 / 0 | 见上表 |
| W2 | Succeeded | 0 / 1 | `mose-width-w2-p1-1x8-1008-0227-3754-36807631` |

W2于20:17:15出现 Training complete，20:17:12最终 step14306 checkpoint 保存成功；20:21:05调度器确认为 Succeeded / active=0 / succeeded=1，可移出未完成监控名单。它的完整 Train CE 尚未在本次操作中汇总。[完成日志](../FFN_Width_Ablations/completion_w2.log)。已完成的五模型 HF 转换任务不计入训练监控。[本次调度快照](MONITOR_STATUS.json)。

## 验证记录

- 三个配置的 meta 参数量、配置序列化、结构注册：6 passed。
- 小模型原生 FFN 公式与反向梯度：3 passed。
- 新增代码 Ruff / Python 编译及 launcher Bash 语法检查通过。
- 新架构已注册统一 HF exporter adapter，三种配置共用一个模块；HF checkpoint 数值验证代码已添加，状态仍为 `not_run`。


## 2026-10-08 23:55 HKT — 8条训练当前进度

本轮核验：5条已完成，3条运行中，无排队任务。五条完成任务均Succeeded/active=0，step14306最终checkpoint保存成功且Training complete。当前三条Dense FFN各为arch pool 1×8 GPU、priority1，共24 GPU。

| 任务 | 状态 | 最新step / 14306 | 最新日志CE（对应step） | 剩余时间，日志估计 |
| --- | --- | ---: | --- | --- |
| W1 | Succeeded | 14306 | 2.504（step14300） | 已完成 |
| W1-QK | Succeeded | 14306 | 未在本轮汇总 | 已完成 |
| W1-M | Succeeded | 14306 | 未在本轮汇总 | 已完成 |
| W2 | Succeeded | 14306 | 未在本轮汇总 | 已完成 |
| G1 | Succeeded | 14306 | 未在本轮汇总 | 已完成 |
| DU | Running | 3039 | 2.736（step3030） | 12h39m |
| DG | Running | 2999 | 2.722（step2990） | 12h41m |
| DUG | Running | 2996 | 2.735（step2990） | 12h44m |

三条运行任务的最近日志尾部未见OOM、NaN/Inf、Traceback或SIGTERM。日志CE只作为进度记录；本轮未汇总五条完成任务的完整/末1000步Train CE。W1/W1-QK/W1-M均约20:22完成，W2于20:17完成，G1于21:07完成。

[本轮完整状态快照](../Dense_FFN_Linear_RMS/MONITOR_STATUS_20261008_2355.json)。后续未完成训练监控名单缩为DU/DG/DUG三条；已完成的转换及OLMES评测任务不计入。

## 2026-10-09 — 参数量重新核对

本机实际实例化四个完整原生meta模型，sum(p.numel())均与config.model.num_params及jobs.json一致：Dense 1,484,916,736；DU/DG各1,482,819,584（少2,097,152，0.141230%）；DUG 1,481,771,008（少3,145,728，0.211845%）。本次只核对模型结构和参数计数，没有GPU评测或修改正在运行的训练。rank1632向下取整导致近似匹配，不能称严格同参数。[参数审计](PARAMETER_AUDIT_20261009.json)。

## 2026-10-08T20:43:59.876322+00:00 — 半小时监控

| 任务 | 状态 | 最新step / 14306 | 最近日志CE（对应step） | 日志ETA | 完成证据 |
| --- | --- | ---: | --- | --- | --- |
| DU | Running | 7347 | 2.528（7340） | 7h47m | 尚未完成 |
| DG | Running | 7310 | 2.515（7310） | 7h51m | 尚未完成 |
| DUG | Running | 7279 | 2.546（7270） | 7h56m | 尚未完成 |

本次记录：[snapshot](monitor_snapshots/monitor_20261009_0441/snapshot.json)；下一次检查2026-10-09T05:11:00+08:00。日志CE仅作进度记录。

## 2026-10-08T21:12:24.905035+00:00 — 半小时监控

| 任务 | 状态 | 最新step / 14306 | 最近日志CE（对应step） | 日志ETA | 完成证据 |
| --- | --- | ---: | --- | --- | --- |
| DU | Running | 7800 | 2.543（7800） | 7h16m | 尚未完成 |
| DG | Running | 7760 | 2.484（7760） | 7h21m | 尚未完成 |
| DUG | Running | 7728 | 2.534（7720） | 7h25m | 尚未完成 |

本次记录：[snapshot](monitor_snapshots/monitor_20261009_0511/snapshot.json)；下一次检查2026-10-09T05:41:00+08:00。日志CE仅作进度记录。

## 2026-10-08T21:42:49.224648+00:00 — 半小时监控

| 任务 | 状态 | 最新step / 14306 | 最近日志CE（对应step） | 日志ETA | 完成证据 |
| --- | --- | ---: | --- | --- | --- |
| DU | Running | 8253 | 2.485（8250） | 6h46m | 尚未完成 |
| DG | Running | 8211 | 2.534（8210） | 6h51m | 尚未完成 |
| DUG | Running | 8177 | 2.547（8170） | 6h55m | 尚未完成 |

本次记录：[snapshot](monitor_snapshots/monitor_20261009_0541/snapshot.json)；下一次检查2026-10-09T06:11:00+08:00。日志CE仅作进度记录。

## 2026-10-08T22:12:49.454636+00:00 — 半小时监控

| 任务 | 状态 | 最新step / 14306 | 最近日志CE（对应step） | 日志ETA | 完成证据 |
| --- | --- | ---: | --- | --- | --- |
| DU | Running | 8699 | 2.58（8690） | 6h16m | 尚未完成 |
| DG | Running | 8655 | 2.511（8650） | 6h21m | 尚未完成 |
| DUG | Running | 8620 | 2.527（8620） | 6h25m | 尚未完成 |

本次记录：[snapshot](monitor_snapshots/monitor_20261009_0611/snapshot.json)；下一次检查2026-10-09T06:41:00+08:00。日志CE仅作进度记录。

## 2026-10-08T22:42:36.905121+00:00 — 半小时监控

| 任务 | 状态 | 最新step / 14306 | 最近日志CE（对应step） | 日志ETA | 完成证据 |
| --- | --- | ---: | --- | --- | --- |
| DU | Running | 9142 | 2.58（9140） | 5h46m | 尚未完成 |
| DG | Running | 9096 | 2.483（9090） | 5h52m | 尚未完成 |
| DUG | Running | 9059 | 2.525（9050） | 5h56m | 尚未完成 |

本次记录：[snapshot](monitor_snapshots/monitor_20261009_0641/snapshot.json)；下一次检查2026-10-09T07:11:00+08:00。日志CE仅作进度记录。

## 2026-10-08T23:12:29.213857+00:00 — 半小时监控

| 任务 | 状态 | 最新step / 14306 | 最近日志CE（对应step） | 日志ETA | 完成证据 |
| --- | --- | ---: | --- | --- | --- |
| DU | Running | 9587 | 2.594（9580） | 5h17m | 尚未完成 |
| DG | Running | 9538 | 2.513（9530） | 5h22m | 尚未完成 |
| DUG | Running | 9500 | 2.543（9500） | 5h25m | 尚未完成 |

本次记录：[snapshot](monitor_snapshots/monitor_20261009_0711/snapshot.json)；下一次检查2026-10-09T07:41:00+08:00。日志CE仅作进度记录。

## 2026-10-08T23:42:33.648151+00:00 — 半小时监控

| 任务 | 状态 | 最新step / 14306 | 最近日志CE（对应step） | 日志ETA | 完成证据 |
| --- | --- | ---: | --- | --- | --- |
| DU | Running | 10035 | 2.497（10030） | 4h47m | 尚未完成 |
| DG | Running | 9999 | 2.519（9990） | 4h50m | 尚未完成 |
| DUG | Running | 9969 | 2.524（9960） | 4h54m | 尚未完成 |

本次记录：[snapshot](monitor_snapshots/monitor_20261009_0741/snapshot.json)；下一次检查2026-10-09T08:11:00+08:00。日志CE仅作进度记录。

## 2026-10-09T00:13:09.745954+00:00 — 半小时监控

| 任务 | 状态 | 最新step / 14306 | 最近日志CE（对应step） | 日志ETA | 完成证据 |
| --- | --- | ---: | --- | --- | --- |
| DU | Running | 10499 | 2.523（None） | None | 尚未完成 |
| DG | Running | 10461 | 2.432（10460） | 4h19m | 尚未完成 |
| DUG | Running | 10420 | 2.511（10420） | 4h23m | 尚未完成 |

本次记录：[snapshot](monitor_snapshots/monitor_20261009_0811/snapshot.json)；下一次检查2026-10-09T08:41:00+08:00。日志CE仅作进度记录。

## 2026-10-09T00:14:21.468267+00:00 — 半小时监控

| 任务 | 状态 | 最新step / 14306 | 最近日志CE（对应step） | 日志ETA | 完成证据 |
| --- | --- | ---: | --- | --- | --- |
| DU | Running | 10516 | 2.496（10510） | 4h15m | 尚未完成 |
| DG | Running | 10461 | 2.432（10460） | 4h19m | 尚未完成 |
| DUG | Running | 10420 | 2.511（10420） | 4h23m | 尚未完成 |

本次记录：[snapshot](monitor_snapshots/monitor_20261009_0811_metrics_retry/snapshot.json)；下一次检查2026-10-09T08:41:00+08:00。日志CE仅作进度记录。

## 2026-10-09T00:42:19.443505+00:00 — 半小时监控

| 任务 | 状态 | 最新step / 14306 | 最近日志CE（对应step） | 日志ETA | 完成证据 |
| --- | --- | ---: | --- | --- | --- |
| DU | Running | 10947 | 2.508（10940） | 3h45m | 尚未完成 |
| DG | Running | 10892 | 2.473（10890） | 3h50m | 尚未完成 |
| DUG | Running | 10849 | 2.543（10840） | 3h54m | 尚未完成 |

本次记录：[snapshot](monitor_snapshots/monitor_20261009_0841/snapshot.json)；下一次检查2026-10-09T09:11:00+08:00。日志CE仅作进度记录。

## 2026-10-09T01:12:43.999077+00:00 — 半小时监控

| 任务 | 状态 | 最新step / 14306 | 最近日志CE（对应step） | 日志ETA | 完成证据 |
| --- | --- | ---: | --- | --- | --- |
| DU | Running | 11401 | 2.516（11400） | 3h15m | 尚未完成 |
| DG | Running | 11343 | 2.529（11340） | 3h20m | 尚未完成 |
| DUG | Running | 11298 | 2.496（11290） | 3h24m | 尚未完成 |

本次记录：[snapshot](monitor_snapshots/monitor_20261009_0911/snapshot.json)；下一次检查2026-10-09T09:41:00+08:00。日志CE仅作进度记录。

## 2026-10-09T01:43:09.783169+00:00 — 半小时监控

| 任务 | 状态 | 最新step / 14306 | 最近日志CE（对应step） | 日志ETA | 完成证据 |
| --- | --- | ---: | --- | --- | --- |
| DU | Running | 11854 | 2.521（11850） | 2h45m | 尚未完成 |
| DG | Running | 11793 | 2.482（11790） | 2h50m | 尚未完成 |
| DUG | Running | 11747 | 2.522（11740） | 2h54m | 尚未完成 |

本次记录：[snapshot](monitor_snapshots/monitor_20261009_0941/snapshot.json)；下一次检查2026-10-09T10:11:00+08:00。日志CE仅作进度记录。

## 2026-10-09T02:13:51.367053+00:00 — 半小时监控

| 任务 | 状态 | 最新step / 14306 | 最近日志CE（对应step） | 日志ETA | 完成证据 |
| --- | --- | ---: | --- | --- | --- |
| DU | Running | 12312 | 2.476（12310） | 2h14m | 尚未完成 |
| DG | Running | 12249 | 2.473（12240） | 2h19m | 尚未完成 |
| DUG | Running | 12201 | 2.535（12200） | 2h23m | 尚未完成 |

本次记录：[snapshot](monitor_snapshots/monitor_20261009_1011/snapshot.json)；下一次检查2026-10-09T10:41:00+08:00。日志CE仅作进度记录。

## 2026-10-09T02:43:23.001485+00:00 — 半小时监控

| 任务 | 状态 | 最新step / 14306 | 最近日志CE（对应step） | 日志ETA | 完成证据 |
| --- | --- | ---: | --- | --- | --- |
| DU | Running | 12735 | 2.439（12730） | 1h46m | 尚未完成 |
| DG | Running | 12669 | 2.459（12660） | 1h51m | 尚未完成 |
| DUG | Running | 12619 | 2.46（12610） | 1h55m | 尚未完成 |

本次记录：[snapshot](monitor_snapshots/monitor_20261009_1041/snapshot.json)；下一次检查2026-10-09T11:11:00+08:00。日志CE仅作进度记录。

## 2026-10-09T03:13:44.833408+00:00 — 半小时监控

| 任务 | 状态 | 最新step / 14306 | 最近日志CE（对应step） | 日志ETA | 完成证据 |
| --- | --- | ---: | --- | --- | --- |
| DU | Running | 13185 | 2.394（13180） | 1h16m | 尚未完成 |
| DG | Running | 13118 | 2.455（13110） | 1h21m | 尚未完成 |
| DUG | Running | 13065 | 2.4（13060） | 1h25m | 尚未完成 |

本次记录：[snapshot](monitor_snapshots/monitor_20261009_1111/snapshot.json)；下一次检查2026-10-09T11:41:00+08:00。日志CE仅作进度记录。

## 2026-10-09T03:43:43.824603+00:00 — 半小时监控

本次调度状态有效；日志命令参数缺空格，未取得训练指标。紧接着的1141_metrics_retry已重新读取日志并补齐，None不表示训练停止。

| 任务 | 状态 | 最新step / 14306 | 最近日志CE（对应step） | 日志ETA | 完成证据 |
| --- | --- | ---: | --- | --- | --- |
| DU | Running | None | None（None） | None | 尚未完成 |
| DG | Running | None | None（None） | None | 尚未完成 |
| DUG | Running | None | None（None） | None | 尚未完成 |

本次记录：[snapshot](monitor_snapshots/monitor_20261009_1141/snapshot.json)；下一次检查2026-10-09T12:11:00+08:00。日志CE仅作进度记录。

## 2026-10-09T03:44:27.876127+00:00 — 半小时监控

| 任务 | 状态 | 最新step / 14306 | 最近日志CE（对应step） | 日志ETA | 完成证据 |
| --- | --- | ---: | --- | --- | --- |
| DU | Running | 13644 | 2.371（13640） | 45m | 尚未完成 |
| DG | Running | 13573 | 2.397（13570） | 50m | 尚未完成 |
| DUG | Running | 13516 | 2.426（13510） | 54m | 尚未完成 |

本次记录：[snapshot](monitor_snapshots/monitor_20261009_1141_metrics_retry/snapshot.json)；下一次检查2026-10-09T12:11:00+08:00。日志CE仅作进度记录。

## 2026-10-09T04:12:10.820189+00:00 — 半小时监控

| 任务 | 状态 | 最新step / 14306 | 最近日志CE（对应step） | 日志ETA | 完成证据 |
| --- | --- | ---: | --- | --- | --- |
| DU | Running | 14050 | 2.365（14050） | 18m | 尚未完成 |
| DG | Running | 13999 | 2.419（13990） | 22m | 尚未完成 |
| DUG | Running | 13946 | 2.36（13940） | 25m | 尚未完成 |

本次记录：[snapshot](monitor_snapshots/monitor_20261009_1211/snapshot.json)；下一次检查2026-10-09T12:41:00+08:00。日志CE仅作进度记录。

## 2026-10-09T04:42:29.398423+00:00 — 半小时监控

| 任务 | 状态 | 最新step / 14306 | 最近日志CE（对应step） | 日志ETA | 完成证据 |
| --- | --- | ---: | --- | --- | --- |
| DU | Succeeded | 14306 | 2.4（14300） | 1m | 最终checkpoint及训练完成已核实 |
| DG | Succeeded | 14306 | 2.394（14300） | 1m | 最终checkpoint及训练完成已核实 |
| DUG | Succeeded | 14306 | 2.401（14300） | 1m | 最终checkpoint及训练完成已核实 |

本次记录：[snapshot](monitor_snapshots/monitor_20261009_1241/snapshot.json)；三条训练全部完成，半小时监控结束。日志CE仅作进度记录。

## 2026-10-09 12:42 +08:00 — 全部完成

三条任务均在arch pool使用1×8卡、低优先级训练至30B、step14306。调度器均为Succeeded、active=0、succeeded=1；已逐条核实最终checkpoint的保存开始、保存成功及Training complete日志。

| 任务 | 最终checkpoint保存成功（+08:00） | 训练完成（+08:00） | 日志证据 |
| --- | --- | --- | --- |
| DU | 12:28:06 | 12:28:09 | [日志](monitor_snapshots/monitor_20261009_1241/du.log) |
| DG | 12:32:59 | 12:33:02 | [日志](monitor_snapshots/monitor_20261009_1241/dg.log) |
| DUG | 12:36:56 | 12:36:59 | [日志](monitor_snapshots/monitor_20261009_1241/dug.log) |

最终checkpoint路径、调度状态及完成时间见[完成快照](monitor_snapshots/monitor_20261009_1241/snapshot.json)。[监控状态](TRAINING_MONITOR_20261009.json)已标记complete，next_check_at=null，后续半小时轮询结束。

此前14份OLMES评测也已全部完成，每份192项、共2688项；[完整分数和分析](../../eval/OLMES_Easy/results/rank512_c0_c8_20261009/METRIC_SUMMARY.md)。DU/DG/DUG不属于这14份评测，不将其训练日志CE混入benchmark结果。

## DU W&B同步完成 — 2026-10-09T05:28:01.154Z

已通过wandb_sjy上传完整0–14306步训练记录，精确离线段的Finished syncing及.wandb.synced标记均已核实。

[W&B实验](https://wandb.ai/archspace/MoSE/runs/olmo3-1b-dense-nearmatched-up-linear-r1632-down-rms-r1632-fp8-30b)／[同步证据](WANDB_SYNC_DU_20261009.json)。

## 三条Dense FFN W&B均已同步完成

- DU: [olmo3-1b-dense-nearmatched-up-linear-r1632-down-rms-r1632-fp8-30b](https://wandb.ai/archspace/MoSE/runs/olmo3-1b-dense-nearmatched-up-linear-r1632-down-rms-r1632-fp8-30b)；完整0–14306步，Finished syncing及.synced标记已核实。
- DG: [olmo3-1b-dense-nearmatched-gate-linear-r1632-down-rms-r1632-fp8-30b](https://wandb.ai/archspace/MoSE/runs/olmo3-1b-dense-nearmatched-gate-linear-r1632-down-rms-r1632-fp8-30b)；完整0–14306步，Finished syncing及.synced标记已核实。
- DUG: [olmo3-1b-dense-nearmatched-up-gate-linear-r1632-down-rms-r1632-fp8-30b](https://wandb.ai/archspace/MoSE/runs/olmo3-1b-dense-nearmatched-up-gate-linear-r1632-down-rms-r1632-fp8-30b)；完整0–14306步，Finished syncing及.synced标记已核实。
