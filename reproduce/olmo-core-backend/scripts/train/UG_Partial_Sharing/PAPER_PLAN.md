# 七天论文工作计划（2026-10-06 至 2026-10-13）

## 当前问题与可检验假设

目标预算以全部独立rank512为参照：797,050,880参数；大dense
1,484,916,736是质量上限参照。主要选型指标是相同步数的最后1000步
平均Train CE。论文仍需要验证集、下游和真实吞吐/显存。

1. 共享有助于节省UG输入投影预算，但完全共享可能限制分支独立性。
   五个P配置检验共享比例；只有P1因向下取整少0.329%参数。
2. 统一rank512可能不是合适的分配：UG经过原始SwiGLU非线性后，
   Down的线性输出瓶颈可能更需要预算。F1/F2在相同FFN预算下把UG rank
   换给Down；F3是共享与重分配的组合。它们都是严格同参数。
3. 中间SiLU是否有用取决于投影位置，不能由一张归一化热力图推出。
   旧图每token分别除以自己的RMS，无gain的RMSNorm因此按构造几乎不改变
   方向。后续机制分析必须包含原始尺度、B之后的变化，以及尺度匹配对照。

原始SwiGLU SiLU保留。初始八组Q/K/V/O都是独立rank512 Identity；Down都是
RMSNorm无可学习gain；全30B WSD日程，init/data seed12536/34521。
配置表见本目录README和相邻FFN_Rank_Reallocation/README。

新增P0qk/P2qk，把P0/P2的Q/K中间激活改为SiLU，形成严格同参数的
两因素对照；先检验部分共享与QK非线性是否存在组合收益，不根据早期
loss宣布最优。其余配置/seed/预算全部保留。

## 时间安排与分支决策

- 第1天：完成当前八组训练、统一完整历史、训练/验证指标和配置审计。
- 第2天：根据完整30B结果决定主方法，优先解决最大的CE差距。若部分
  共享或预算重分配没有可靠收益，论文采用实证研究定位，不宣称新SOTA。
- 第2–4天：最强候选与匹配对照补种子；目标关键比较达到3个种子。
  对胜出结构再检验Q/K中间SiLU，保持共享/rank/Down和预算完全一致。
- 第3天冻结主要架构；第4–5天补一个次级预算或第二标准模型配置的
  匹配对照，取决于算力和已验证收益。主表仍使用rank512预算。
- 第5–6天：机制、验证/下游、相同硬件效率测试与最终图表。
- 第7天：完成写作、来源核对、复现说明和代码整理。

部分共享、低秩投影和rank分配均有相关工作，不能仅凭结构形式宣称
首创。贡献需要来自具体投影角色的机制、固定预算下的方法收益及完整
对照。单种子约0.001CE变化不作为稳定优越性的证据。

## 结果与复现要求

- 训练任务完成需scheduler终态active0、14306步与最终checkpoint日志。
- 每个尝试使用精确W&B段；停止尝试先同步，再授权重提；排队不重复提。
- 离线历史保留原始异常值，主报告包含原始统计和共同步数敏感性分析。
- 验证用相同11来源、共同最后验证步数的等权macro CE，不把它写成
  token加权平均或BPB。Train CE、平滑值、单个最终值明确区分。
- 多段恢复按实际checkpoint边界/时间顺序拼接，记录重叠步数与来源。
- seed、参数量、实际GPU数、global/microbatch、数据与调度配置全部保存。
- 旧BF16/FP8的7.5B/500warmup实验独立报告，不能充当30B/2000warmup
  日程下7.5B时刻的对照。
- 当前统一监控每1800秒；arch有空16/32卡可供后续新任务，已经运行
  的pool任务继续原处，避免重复写同一checkpoint目录。

## 已完成结果入口

- ../Rank512Matched_Combinations/C0_C7_RESULTS.md
- ../Rank512Matched_Combinations/C8_RESULTS.md
- ../Dense_Single_SiLU/RESULTS.md
- ../Dense_Sharing/RESULTS.md（新增11来源验证结果）

新比较脚本compare_offline.py接受本地W&B文件manifest，同时输出原始
Train CE、共同异常步数处理后的CE及匹配验证macro CE。它已在完成的
Dense/S1/S2/S3历史上复核，所有旧full/tail/final Train CE相差小于1e-12。

## 批次管理（用户2026-10-06新要求）

当前批次固定10组：P0–P4、F1–F3、P0qk/P2qk。先集中启动和恢复这些配置，然后每30分钟统一监控。整个批次都完成30B训练、同步并比较完整Train CE后，才释放下一批（候选组合/重复种子）。运行中的健康任务不为新增实验迁移或停止。checkpoint配额问题允许更换输出存储并显式加载已验证的旧checkpoint；不删除旧checkpoint，不改训练预算/架构/日程。

## 19:19 HKT用户更新

用户要求“有卡直接交就行了”，覆盖此前等待整批完成才提交的限制。允许有空卡立即提交已验证的补充组合；保持健康任务原处运行、低优1、优先arch16/32、排队则pool8。原十组结果仍作为固定第一张完整对照表，batch_segments.json不混入新任务。下一组补F1/F2/F3的QK-SiLU对照，其余配置/seed/797,050,880参数预算不变；原始SwiGLU与Down RMSNorm保持。统一30min监控继续覆盖新任务。

## 21:06 HKT prepared seed confirmation

C0/C6 paired initialization-seed12537/12538 replications are prepared under ../Rank512_Seed_Replication. Data seed remains34521; architecture and full30B recipe match original config builders exactly. Actual meta-module budgets and 8/16/32-card global-batch divisibility passed the dedicated test (1passed); all launchers passed bash-n. Pending capacity, not submitted at preparation time. Original C6 single-seed tail CE gain over C0 is only0.004215, so these replications test whether that gain survives different initialization. New F1qk/F2qk/F3qk runs continue normally.

## 21:11 HKT — Q/K SiLU sharing-ratio sweep

Prepared and tested P3qk/P4qk, changing only Q/K intermediate Identity to SiLU versus P3/P4. These complete the exact-parameter P0/P2/P3/P4 sharing-ratio sweep under both Q/K nonlinearities; P1 remains a slightly smaller, separately marked endpoint. Dedicated config-equality and actual meta-parameter test passed across all four QK variants (1passed); new wrappers passed bash-n. New submissions use available pool1x8, low priority1. Main original-ten manifest remains unchanged.

## 21:35 HKT — full-history cohort reporting

The exporter now accepts explicitly completed subsets and the full19-config cohort while retaining the fixed original-ten gate by default. Every exported config must match current job ID/latest exact segment, final14306/Succeeded/active0, and saved final checkpoint; actual history must contain every optimizer step1..14306. It preserves raw anomalous records, computes common-mask Train CE, records allocations/seeds/resume boundaries, and separates strict797,050,880-param configs from P1. Q/K effects across P/F pairs and matched initialization-seed differences are added. Original seed12536 is not silently merged into new seed statistics with a different anomaly mask. Six guard/history tests passed; export wrapper bash-n passed. Prepared cohort_segments.json for all19 and a completed P2/P2qk subset for early exact comparison on an available scheduled pool8 allocation.

## 23:08 HKT — completed F3qk and fixed-sharing allocation controls

F3qk completed14306/checkpoint/Succeeded/active0 and W&B finished syncing.
An authorized scheduled read-only arch8 report recomputed P2/P2qk/F3qk and
archived C0/C2/C6 with one common anomaly mask. The recovered full JSON matches
the emitted SHA256. C6 remains best tail2.474762967, followed by C2
2.475667983 and F3qk2.477421343. The historical mask differs from the original
eight-run table because step14304 from C3 is not present in this cohort.
See F3QK_WITH_REFERENCES_RESULTS.md; historical references use archived
completion evidence, not invented fresh scheduler observations.

F4qk(sharedUG400/Down816) and F5qk(sharedUG560/Down528) now run on available
arch2x8 priority1. They keep sharing scope and Q/K SiLU fixed versus F3qk,
forming a strictly797,050,880-parameter allocation curve. Actual meta counts,
recipe equality and8/16/32-card global-batch checks passed2 tests; both startup
meshes, step0 saves and finite training verified23:08. Training cohort now21;
the original-ten batch manifest remains unchanged.

C2s2/C2s3 are prepared and locally verified as exact C2 architecture replications
at init12537/12538 with fixed data34521. They remain unsubmitted until matching
pool8 capacity is available. They separate the C2-minus-C0 sharing/rank recipe
from C6-minus-C2 Q-SiLU. The expanded seed configuration test passed; the
completed exporter now reports three paired contrasts and rejects unmatched
seeds/allocations/counts. Ten cohort guard/history tests and three archived
reference integrity tests passed. Do not aggregate original12536 into seed
statistics until its corresponding metadata and common history are verified.

## 23:43 HKT — C2 controls and monitoring cadence

P0 completed14306 at23:25 with final checkpoint saved, Succeeded/active0,
and its exact W&B upload finished. C2s2/C2s3 submitted to newly available
pool1x8, priority1, at23:38/23:40. Both fresh step0 saves and mesh(1,8)
verified; C2s2 first finite CE observed, C2s3 first CE pending this snapshot.
The current training registry/cohort has23 configs: four completed and19
Running, including new launches. Original-ten manifest stays fixed.

The local monitor was intentionally reloaded after the completed23:33 poll:
next start23:56, then every30 minutes measured between poll starts, subtracting
query duration. Routine logs use the latest200 lines; startup checks can request
5000. GPU training jobs were not restarted. Report reconstruction now joins
scheduler transport chunks only when their full SHA256 matches, with five
integrity tests passed and the real F3qk report independently recovered.
