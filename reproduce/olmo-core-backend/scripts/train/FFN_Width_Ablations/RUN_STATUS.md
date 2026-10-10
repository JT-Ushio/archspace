# FFN宽度与瓶颈实验

## 2026-10-08 — Prepared and locally validated

W1(C2, m6144/sharedUG736/Down640) and W2(C2, m4096/sharedUG976/Down944) each have797,050,880 real meta-model parameters; per-layer FFN15,794,176. Optional G1 retainsC2(m8192/sharedUG568/Down520) and adds32 adaptiveQ/K parameters, total797,050,912. Globalbatch256/seq8192/30B/Muon.005/WSD2000+.2/FP8/checkpoint1788 preserved.

11 CPU/meta checks passed in91.55s, with expected Flash3-to-Torch fallback warnings because local GPUs are unavailable. Actualwidth FFN forward/backward, initial C2 function/input-gradient equivalence, gate training/checkpoint restoration, AdamW/no-decay categorization, compile capture and two-rank empty-shard metric reconstruction verified. New files ruff passed; all four launchers bash-n passed. [Audit](local_validation.json) / [Detailed design](EXPERIMENT_DESIGN.md).

All three variants prepared_not_submitted. No GPU training, scheduler query, submission, W&B run creation or monitor loop performed for these variants. Existing completed experiments and original model files retained.

## 2026-10-08 — Controls added; submission held by user

用户最新指令为“先不要提交”。五条配置均仅准备，submission_plan.json逐条标记submission_allowed=false。没有GPU提交、调度查询、W&B run创建或新的监控循环。

| 配置 | FFN宽度 / UG共享rank / Down rank | Q/K中间变换 | 全模型参数 | 状态 |
| --- | --- | --- | ---: | --- |
| W1 | 6144 / 736 / 640 | Linear / SiLU | 797,050,880 | prepared_not_submitted |
| W1-QK | 6144 / 736 / 640 | SiLU / SiLU | 797,050,880 | prepared_not_submitted |
| W1-M | 6144 / 568 / 520 | Linear / SiLU | 742,787,072 | prepared_not_submitted |
| W2 | 4096 / 976 / 944 | Linear / SiLU | 797,050,880 | prepared_not_submitted |
| G1 | 8192 / 568 / 520 | 学习alpha，初始0 / 1 | 797,050,912 | prepared_not_submitted，备选 |

所有配置Down无gain RMSNorm、UG中间Linear并保留原生SwiGLU。W1-M是故意降低预算的缩宽对照，不能归入严格同参数实验。恢复提交后的候选顺序为W1、W1-QK、W1-M、W2、G1。

本次9项宽度/控制组CPU/meta测试通过，耗时19.13s；5条预期CPU Attention backend fallback警告。真实全模型计数、配方、Q激活与W1-M实际前向反向已验证。六个shell脚本bash-n通过，新增及改动的配置/审计/测试ruff通过，文档表格及相对链接通过。旧G1实现未改动，其此前检查证据保留。

W1/W1-QK原生完整16层CPU FP32初始化比较于02:20:31 HKT完成：275张参数、797,050,880个元素逐张torch.equal全相等，有限值检查通过；注册buffer为0。仅核对FP8/FSDP前的CPU权重，不称实际GPU step0已验证。[逐张结果](initialization_validation.json)。

本次明确区分seed字段：运行级config.init_seed=12536、模型权重Generator的model.init_seed=0、data seed=34521，沿用C2实际配置。独立权重seed重复必须显式修改model.init_seed，不能仅修改运行级seed。设计文档补充W1-QK交叉差分、W1-M缩宽/容量回补解释链、实际GPU step0后续核对要求。

[详细设计](EXPERIMENT_DESIGN.md) / [参数与检查证据](local_validation.json) / [暂不提交计划](submission_plan.json)。

## 2026-10-08T02:25:47.689427+08:00 — User resumed all five submissions and monitoring

用户明确恢复提交W1、W1-QK、W1-M、W2、G1，并要求持续至五条完成。arch优先2×8且不排队；否则pool 1×8（gateway别名archp），PRIORITY=1；每30min查询一次。旧暂缓指令由本条新授权撤销。
W1/W1-QK新增一次性初始权重指纹记录：全部rank共同还原初始参数，仅rank0记录完整参数字节的SHA256与逐张清单，位于step0、dry-run和优化器更新之前，不改变参数或RNG。将从GPU启动日志比较实际初始指纹。其余模型/配方不变。

Pre-submit revalidation: 11 combined width/fingerprint CPU/meta checks passed in72.51s with5 expected CPU backend warnings. Two-rank full hashing including an empty shard, BF16 byte hashing, unchanged weights/RNG and resume skip passed. Ruff passed for new source/config/tests/orchestration.

Submitted W1: `mose-width-w1-p1-1x8-1008-0226-37ef-94920163`, pool 1x8, priority1; script `/mnt/shared-storage-gpfs2/intern-pretrain-shared02/shijiayang-p/archspace-MoSE/reproduce/olmo-core-backend/scripts/train/FFN_Width_Ablations/run_w1.sh`.

Submitted W1QK: `mose-width-w1qk-p1-1x8-1008-0227-b83b-25485220`, pool 1x8, priority1; script `/mnt/shared-storage-gpfs2/intern-pretrain-shared02/shijiayang-p/archspace-MoSE/reproduce/olmo-core-backend/scripts/train/FFN_Width_Ablations/run_w1qk.sh`.

Submitted W1M: `mose-width-w1m-p1-1x8-1008-0227-cf64-30995519`, pool 1x8, priority1; script `/mnt/shared-storage-gpfs2/intern-pretrain-shared02/shijiayang-p/archspace-MoSE/reproduce/olmo-core-backend/scripts/train/FFN_Width_Ablations/run_w1m.sh`.

Submitted W2: `mose-width-w2-p1-1x8-1008-0227-3754-36807631`, pool 1x8, priority1; script `/mnt/shared-storage-gpfs2/intern-pretrain-shared02/shijiayang-p/archspace-MoSE/reproduce/olmo-core-backend/scripts/train/FFN_Width_Ablations/run_w2.sh`.

Submitted G1: `mose-width-g1-p1-1x8-1008-0227-1760-42442674`, pool 1x8, priority1; script `/mnt/shared-storage-gpfs2/intern-pretrain-shared02/shijiayang-p/archspace-MoSE/reproduce/olmo-core-backend/scripts/train/FFN_Width_Ablations/run_g1.sh`.

## 2026-10-08 02:27 HKT — monitoring

- w1: `mose-width-w1-p1-1x8-1008-0226-37ef-94920163`; Inqueue; {'active': 0, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
- w1qk: `mose-width-w1qk-p1-1x8-1008-0227-b83b-25485220`; Inqueue; {'active': 0, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
- w1m: `mose-width-w1m-p1-1x8-1008-0227-cf64-30995519`; Inqueue; {'active': 0, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
- w2: `mose-width-w2-p1-1x8-1008-0227-3754-36807631`; Inqueue; {'active': 0, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
- g1: `mose-width-g1-p1-1x8-1008-0227-1760-42442674`; Inqueue; {'active': 0, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.

## 2026-10-08 02:28 HKT — Five confirmed jobs; first observation

W1、W1-QK、W1-M、W2、G1均已提交至pool（archp）1×8、PRIORITY=1。02:27:58调度器五条均Inqueue，active=0，尚无训练步数和W&B段；arch2×8资源预测不可即时调度。新任务不因排队重提。
第一次rjob logs查询因replica尚无日志而在gateway CLI的brainpp/rjob中报NoneType，属于日志读取失败，不是训练Traceback；已给本地监控解析器加分类。只重启本地观察程序，未停止/修改任何GPU任务，保留30分钟截止时间，下一轮约02:57:58 HKT。

## 2026-10-08 02:58 HKT — monitoring

- w1: `mose-width-w1-p1-1x8-1008-0226-37ef-94920163`; Inqueue; {'active': 0, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
- w1qk: `mose-width-w1qk-p1-1x8-1008-0227-b83b-25485220`; Inqueue; {'active': 0, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
- w1m: `mose-width-w1m-p1-1x8-1008-0227-cf64-30995519`; Inqueue; {'active': 0, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
- w2: `mose-width-w2-p1-1x8-1008-0227-3754-36807631`; Inqueue; {'active': 0, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
- g1: `mose-width-g1-p1-1x8-1008-0227-1760-42442674`; Inqueue; {'active': 0, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.

## 2026-10-08 03:28 HKT — monitoring

- w1: `mose-width-w1-p1-1x8-1008-0226-37ef-94920163`; Inqueue; {'active': 0, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
- w1qk: `mose-width-w1qk-p1-1x8-1008-0227-b83b-25485220`; Inqueue; {'active': 0, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
- w1m: `mose-width-w1m-p1-1x8-1008-0227-cf64-30995519`; Inqueue; {'active': 0, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
- w2: `mose-width-w2-p1-1x8-1008-0227-3754-36807631`; Inqueue; {'active': 0, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
- g1: `mose-width-g1-p1-1x8-1008-0227-1760-42442674`; Inqueue; {'active': 0, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.

## 2026-10-08 03:58 HKT — monitoring

- w1: `mose-width-w1-p1-1x8-1008-0226-37ef-94920163`; Inqueue; {'active': 0, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
- w1qk: `mose-width-w1qk-p1-1x8-1008-0227-b83b-25485220`; Inqueue; {'active': 0, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
- w1m: `mose-width-w1m-p1-1x8-1008-0227-cf64-30995519`; Inqueue; {'active': 0, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
- w2: `mose-width-w2-p1-1x8-1008-0227-3754-36807631`; Inqueue; {'active': 0, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
- g1: `mose-width-g1-p1-1x8-1008-0227-1760-42442674`; Inqueue; {'active': 0, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.

## 2026-10-08 04:28 HKT — monitoring

- w1: `mose-width-w1-p1-1x8-1008-0226-37ef-94920163`; Inqueue; {'active': 0, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
- w1qk: `mose-width-w1qk-p1-1x8-1008-0227-b83b-25485220`; Inqueue; {'active': 0, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
- w1m: `mose-width-w1m-p1-1x8-1008-0227-cf64-30995519`; Inqueue; {'active': 0, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
- w2: `mose-width-w2-p1-1x8-1008-0227-3754-36807631`; Inqueue; {'active': 0, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
- g1: `mose-width-g1-p1-1x8-1008-0227-1760-42442674`; Inqueue; {'active': 0, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.

## 2026-10-08 04:58 HKT — monitoring

- w1: `mose-width-w1-p1-1x8-1008-0226-37ef-94920163`; Inqueue; {'active': 0, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
- w1qk: `mose-width-w1qk-p1-1x8-1008-0227-b83b-25485220`; Inqueue; {'active': 0, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
- w1m: `mose-width-w1m-p1-1x8-1008-0227-cf64-30995519`; Inqueue; {'active': 0, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
- w2: `mose-width-w2-p1-1x8-1008-0227-3754-36807631`; Inqueue; {'active': 0, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
- g1: `mose-width-g1-p1-1x8-1008-0227-1760-42442674`; Inqueue; {'active': 0, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.

## 2026-10-08 05:28 HKT — monitoring

- w1: `mose-width-w1-p1-1x8-1008-0226-37ef-94920163`; Inqueue; {'active': 0, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
- w1qk: `mose-width-w1qk-p1-1x8-1008-0227-b83b-25485220`; Inqueue; {'active': 0, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
- w1m: `mose-width-w1m-p1-1x8-1008-0227-cf64-30995519`; Inqueue; {'active': 0, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
- w2: `mose-width-w2-p1-1x8-1008-0227-3754-36807631`; Inqueue; {'active': 0, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
- g1: `mose-width-g1-p1-1x8-1008-0227-1760-42442674`; Inqueue; {'active': 0, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.

## 2026-10-08 05:58 HKT — monitoring

- w1: `mose-width-w1-p1-1x8-1008-0226-37ef-94920163`; Inqueue; {'active': 0, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
- w1qk: `mose-width-w1qk-p1-1x8-1008-0227-b83b-25485220`; Inqueue; {'active': 0, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
- w1m: `mose-width-w1m-p1-1x8-1008-0227-cf64-30995519`; Inqueue; {'active': 0, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
- w2: `mose-width-w2-p1-1x8-1008-0227-3754-36807631`; Inqueue; {'active': 0, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
- g1: `mose-width-g1-p1-1x8-1008-0227-1760-42442674`; Inqueue; {'active': 0, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.

## 2026-10-08 06:28 HKT — monitoring

- w1: `mose-width-w1-p1-1x8-1008-0226-37ef-94920163`; Inqueue; {'active': 0, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
- w1qk: `mose-width-w1qk-p1-1x8-1008-0227-b83b-25485220`; Inqueue; {'active': 0, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
- w1m: `mose-width-w1m-p1-1x8-1008-0227-cf64-30995519`; Inqueue; {'active': 0, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
- w2: `mose-width-w2-p1-1x8-1008-0227-3754-36807631`; Inqueue; {'active': 0, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
- g1: `mose-width-g1-p1-1x8-1008-0227-1760-42442674`; Inqueue; {'active': 0, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.

## 2026-10-08 06:58 HKT — monitoring

- w1: `mose-width-w1-p1-1x8-1008-0226-37ef-94920163`; Inqueue; {'active': 0, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
- w1qk: `mose-width-w1qk-p1-1x8-1008-0227-b83b-25485220`; Inqueue; {'active': 0, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
- w1m: `mose-width-w1m-p1-1x8-1008-0227-cf64-30995519`; Inqueue; {'active': 0, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
- w2: `mose-width-w2-p1-1x8-1008-0227-3754-36807631`; Inqueue; {'active': 0, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
- g1: `mose-width-g1-p1-1x8-1008-0227-1760-42442674`; Inqueue; {'active': 0, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.

## 2026-10-08 07:28 HKT — monitoring

- w1: `mose-width-w1-p1-1x8-1008-0226-37ef-94920163`; Inqueue; {'active': 0, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
- w1qk: `mose-width-w1qk-p1-1x8-1008-0227-b83b-25485220`; Inqueue; {'active': 0, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
- w1m: `mose-width-w1m-p1-1x8-1008-0227-cf64-30995519`; Inqueue; {'active': 0, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
- w2: `mose-width-w2-p1-1x8-1008-0227-3754-36807631`; Inqueue; {'active': 0, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
- g1: `mose-width-g1-p1-1x8-1008-0227-1760-42442674`; Inqueue; {'active': 0, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.

## 2026-10-08 07:30 HKT — Prediction validity check

真实五条job仍Inqueue、active=0。复查pool预测原始输出发现请求1副本却返回总副本数量0，因此不能把该返回值解读为pool没有空卡；capacity.py已增加副本数一致性检查，无效预测记为schedulable=null。arch请求2副本且返回2，预测有效、当前仍不可立即调度。仅重启本地观察进程以载入检查，五条GPU任务及资源/超参保持原状，下一轮沿用30分钟截止时间。

## 2026-10-08 07:58 HKT — monitoring

- w1: `mose-width-w1-p1-1x8-1008-0226-37ef-94920163`; Inqueue; {'active': 0, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
- w1qk: `mose-width-w1qk-p1-1x8-1008-0227-b83b-25485220`; Inqueue; {'active': 0, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
- w1m: `mose-width-w1m-p1-1x8-1008-0227-cf64-30995519`; Inqueue; {'active': 0, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
- w2: `mose-width-w2-p1-1x8-1008-0227-3754-36807631`; Inqueue; {'active': 0, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
- g1: `mose-width-g1-p1-1x8-1008-0227-1760-42442674`; Inqueue; {'active': 0, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.

## 2026-10-08 08:28 HKT — monitoring

- w1: `mose-width-w1-p1-1x8-1008-0226-37ef-94920163`; Inqueue; {'active': 0, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
- w1qk: `mose-width-w1qk-p1-1x8-1008-0227-b83b-25485220`; Inqueue; {'active': 0, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
- w1m: `mose-width-w1m-p1-1x8-1008-0227-cf64-30995519`; Inqueue; {'active': 0, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
- w2: `mose-width-w2-p1-1x8-1008-0227-3754-36807631`; Inqueue; {'active': 0, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
- g1: `mose-width-g1-p1-1x8-1008-0227-1760-42442674`; Inqueue; {'active': 0, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.

## 2026-10-08 08:58 HKT — monitoring

- w1: `mose-width-w1-p1-1x8-1008-0226-37ef-94920163`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 300, 'eta': '10h56m', 'grad': 0.2325, 'ce': 4.137}.
- w1qk: `mose-width-w1qk-p1-1x8-1008-0227-b83b-25485220`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 300, 'eta': '10h56m', 'grad': 0.2376, 'ce': 4.184}.
- w1m: `mose-width-w1m-p1-1x8-1008-0227-cf64-30995519`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 310, 'eta': '10h58m', 'grad': 0.2628, 'ce': 4.119}.
- w2: `mose-width-w2-p1-1x8-1008-0227-3754-36807631`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 310, 'eta': '10h50m', 'grad': 0.2318, 'ce': 4.098}.
- g1: `mose-width-g1-p1-1x8-1008-0227-1760-42442674`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 290, 'eta': '11h44m', 'grad': 0.2853, 'ce': 4.195}.

## 2026-10-08 08:58 HKT — All five running; GPU startup validated

五条约08:40开始启动，均pool 1×8 H200、低优1，8 ranks/dp_replicate1+dp_shard8，从头初始化，step0保存成功、dry-run完成，当前CE/grad均有限且无训练OOM/NaN/Traceback。

- w1: step300, CE4.137, grad0.2325, ETA10h56m; 参数797,050,880。
- w1qk: step300, CE4.184, grad0.2376, ETA10h56m; 参数797,050,880。
- w1m: step310, CE4.119, grad0.2628, ETA10h58m; 参数742,787,072。
- w2: step310, CE4.098, grad0.2318, ETA10h50m; 参数797,050,880。
- g1: step290, CE4.195, grad0.2853, ETA11h44m; 参数797,050,912。

W1/W1-QK实际GPU起始全部275张参数、797,050,880个元素的原始字节SHA256完全一致：96abe17d492e6c7596279f41e90092ebb13eab9eee6e37e7973be9295bdd6952。验证发生于dry-run和首次optimizer更新之前，不将单个末步或早期曲线当作完整质量结论。
五条精确offline W&B段、启动日志和参数数目见[gpu_startup_validation.json](gpu_startup_validation.json)。归档启动证据保持不变，30分钟观察继续。

## 2026-10-08 09:28 HKT — monitoring

- w1: `mose-width-w1-p1-1x8-1008-0226-37ef-94920163`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 910, 'eta': '10h46m', 'grad': 0.1566, 'ce': 3.201}.
- w1qk: `mose-width-w1qk-p1-1x8-1008-0227-b83b-25485220`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 910, 'eta': '10h47m', 'grad': 0.2424, 'ce': 3.209}.
- w1m: `mose-width-w1m-p1-1x8-1008-0227-cf64-30995519`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 920, 'eta': '10h48m', 'grad': 0.2096, 'ce': 3.198}.
- w2: `mose-width-w2-p1-1x8-1008-0227-3754-36807631`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 930, 'eta': '10h40m', 'grad': 0.2002, 'ce': 3.152}.
- g1: `mose-width-g1-p1-1x8-1008-0227-1760-42442674`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 860, 'eta': '11h37m', 'grad': 0.1933, 'ce': 3.206}.

## 2026-10-08 09:58 HKT — monitoring

- w1: `mose-width-w1-p1-1x8-1008-0226-37ef-94920163`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 1510, 'eta': '10h28m', 'grad': 0.1614, 'ce': 2.997}.
- w1qk: `mose-width-w1qk-p1-1x8-1008-0227-b83b-25485220`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 1510, 'eta': '10h29m', 'grad': 0.1301, 'ce': 3.0}.
- w1m: `mose-width-w1m-p1-1x8-1008-0227-cf64-30995519`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 1510, 'eta': '10h31m', 'grad': 0.141, 'ce': 3.01}.
- w2: `mose-width-w2-p1-1x8-1008-0227-3754-36807631`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 1530, 'eta': '10h22m', 'grad': 0.0792, 'ce': 3.064}.
- g1: `mose-width-g1-p1-1x8-1008-0227-1760-42442674`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 1440, 'eta': '11h8m', 'grad': 0.1139, 'ce': 3.038}.

## 2026-10-08 10:28 HKT — monitoring

- w1: `mose-width-w1-p1-1x8-1008-0226-37ef-94920163`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 2130, 'eta': '9h56m', 'grad': 0.1129, 'ce': 2.926}.
- w1qk: `mose-width-w1qk-p1-1x8-1008-0227-b83b-25485220`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 2130, 'eta': '9h56m', 'grad': 0.0787, 'ce': 2.923}.
- w1m: `mose-width-w1m-p1-1x8-1008-0227-cf64-30995519`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 2130, 'eta': '9h58m', 'grad': 0.0664, 'ce': 2.938}.
- w2: `mose-width-w2-p1-1x8-1008-0227-3754-36807631`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 2160, 'eta': '9h49m', 'grad': 0.0569, 'ce': 2.843}.
- g1: `mose-width-g1-p1-1x8-1008-0227-1760-42442674`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 1990, 'eta': '10h41m', 'grad': 0.0692, 'ce': 2.932}.

## 2026-10-08 10:58 HKT — monitoring

- w1: `mose-width-w1-p1-1x8-1008-0226-37ef-94920163`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 2750, 'eta': '9h24m', 'grad': 0.0297, 'ce': 2.775}.
- w1qk: `mose-width-w1qk-p1-1x8-1008-0227-b83b-25485220`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 2750, 'eta': '9h24m', 'grad': 0.0606, 'ce': 2.774}.
- w1m: `mose-width-w1m-p1-1x8-1008-0227-cf64-30995519`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 2750, 'eta': '9h26m', 'grad': 0.0353, 'ce': 2.79}.
- w2: `mose-width-w2-p1-1x8-1008-0227-3754-36807631`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 2780, 'eta': '9h18m', 'grad': 0.0423, 'ce': 2.799}.
- g1: `mose-width-g1-p1-1x8-1008-0227-1760-42442674`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 2570, 'eta': '10h15m', 'grad': 0.035, 'ce': 2.895}.

## 2026-10-08 11:28 HKT — monitoring

- w1: `mose-width-w1-p1-1x8-1008-0226-37ef-94920163`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 3370, 'eta': '8h52m', 'grad': 0.0437, 'ce': 2.822}.
- w1qk: `mose-width-w1qk-p1-1x8-1008-0227-b83b-25485220`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 3370, 'eta': '8h53m', 'grad': 0.0881, 'ce': 2.821}.
- w1m: `mose-width-w1m-p1-1x8-1008-0227-cf64-30995519`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 3370, 'eta': '8h54m', 'grad': 0.04, 'ce': 2.838}.
- w2: `mose-width-w2-p1-1x8-1008-0227-3754-36807631`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 3410, 'eta': '8h46m', 'grad': 0.026, 'ce': 2.813}.
- g1: `mose-width-g1-p1-1x8-1008-0227-1760-42442674`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 3150, 'eta': '9h43m', 'grad': 0.0715, 'ce': 2.859}.

## 2026-10-08 11:58 HKT — monitoring

- w1: `mose-width-w1-p1-1x8-1008-0226-37ef-94920163`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 3990, 'eta': '8h22m', 'grad': 0.0798, 'ce': 2.756}.
- w1qk: `mose-width-w1qk-p1-1x8-1008-0227-b83b-25485220`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 3990, 'eta': '8h22m', 'grad': 0.0775, 'ce': 2.754}.
- w1m: `mose-width-w1m-p1-1x8-1008-0227-cf64-30995519`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 3990, 'eta': '8h24m', 'grad': 0.0322, 'ce': 2.771}.
- w2: `mose-width-w2-p1-1x8-1008-0227-3754-36807631`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 4010, 'eta': '8h20m', 'grad': 0.057, 'ce': 2.716}.
- g1: `mose-width-g1-p1-1x8-1008-0227-1760-42442674`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 3730, 'eta': '9h12m', 'grad': 0.0239, 'ce': 2.672}.

## 2026-10-08 12:28 HKT — monitoring

- w1: `mose-width-w1-p1-1x8-1008-0226-37ef-94920163`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 4590, 'eta': '7h55m', 'grad': 0.03, 'ce': 2.664}.
- w1qk: `mose-width-w1qk-p1-1x8-1008-0227-b83b-25485220`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 4590, 'eta': '7h55m', 'grad': 0.0375, 'ce': 2.662}.
- w1m: `mose-width-w1m-p1-1x8-1008-0227-cf64-30995519`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 4590, 'eta': '7h56m', 'grad': 0.0292, 'ce': 2.679}.
- w2: `mose-width-w2-p1-1x8-1008-0227-3754-36807631`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 4640, 'eta': '7h48m', 'grad': 0.0707, 'ce': 2.736}.
- g1: `mose-width-g1-p1-1x8-1008-0227-1760-42442674`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 4320, 'eta': '8h41m', 'grad': 0.0336, 'ce': 2.725}.

## 2026-10-08 12:58 HKT — monitoring

- w1: `mose-width-w1-p1-1x8-1008-0226-37ef-94920163`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 5210, 'eta': '7h24m', 'grad': 0.0454, 'ce': 2.74}.
- w1qk: `mose-width-w1qk-p1-1x8-1008-0227-b83b-25485220`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 5210, 'eta': '7h24m', 'grad': 0.0484, 'ce': 2.738}.
- w1m: `mose-width-w1m-p1-1x8-1008-0227-cf64-30995519`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 5210, 'eta': '7h25m', 'grad': 0.051, 'ce': 2.754}.
- w2: `mose-width-w2-p1-1x8-1008-0227-3754-36807631`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 5260, 'eta': '7h18m', 'grad': 0.0299, 'ce': 2.748}.
- g1: `mose-width-g1-p1-1x8-1008-0227-1760-42442674`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 4900, 'eta': '8h10m', 'grad': 0.017, 'ce': 2.724}.

## 2026-10-08 13:40 HKT — monitoring

- w1: `mose-width-w1-p1-1x8-1008-0226-37ef-94920163`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 6070, 'eta': '6h42m', 'grad': 0.0975, 'ce': 2.681}.
- w1qk: `mose-width-w1qk-p1-1x8-1008-0227-b83b-25485220`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 6060, 'eta': '6h43m', 'grad': 0.0175, 'ce': 2.749}.
- w1m: `mose-width-w1m-p1-1x8-1008-0227-cf64-30995519`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 6070, 'eta': '6h43m', 'grad': 0.0339, 'ce': 2.696}.
- w2: `mose-width-w2-p1-1x8-1008-0227-3754-36807631`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 6130, 'eta': '6h36m', 'grad': 0.0171, 'ce': 2.7}.
- g1: `mose-width-g1-p1-1x8-1008-0227-1760-42442674`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 5700, 'eta': '7h29m', 'grad': 0.0252, 'ce': 2.604}.

## 2026-10-08 13:40 HKT — local observer recovery

Previous observer PID 259844 and tool session 79546 were both confirmed missing after an interrupted wait. No GPU jobs were stopped or resubmitted. Restarted only the local observer as detached PID 8108; fresh scheduler checks at 13:40 confirmed all five jobs Running with one active replica each and finite CE/gradient. The 13:28 round was missed; next round is approximately 14:10 HKT.

## 2026-10-08 14:10 HKT — monitoring

- w1: `mose-width-w1-p1-1x8-1008-0226-37ef-94920163`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 6690, 'eta': '6h12m', 'grad': 0.0456, 'ce': 2.595}.
- w1qk: `mose-width-w1qk-p1-1x8-1008-0227-b83b-25485220`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 6680, 'eta': '6h12m', 'grad': 0.0292, 'ce': 2.64}.
- w1m: `mose-width-w1m-p1-1x8-1008-0227-cf64-30995519`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 6690, 'eta': '6h13m', 'grad': 0.0319, 'ce': 2.611}.
- w2: `mose-width-w2-p1-1x8-1008-0227-3754-36807631`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 6750, 'eta': '6h6m', 'grad': 0.0615, 'ce': 2.653}.
- g1: `mose-width-g1-p1-1x8-1008-0227-1760-42442674`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 6280, 'eta': '6h58m', 'grad': 0.0277, 'ce': 2.684}.

## 2026-10-08 14:40 HKT — monitoring

- w1: `mose-width-w1-p1-1x8-1008-0226-37ef-94920163`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 7310, 'eta': '5h41m', 'grad': 0.0228, 'ce': 2.62}.
- w1qk: `mose-width-w1qk-p1-1x8-1008-0227-b83b-25485220`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 7300, 'eta': '5h42m', 'grad': 0.014, 'ce': 2.651}.
- w1m: `mose-width-w1m-p1-1x8-1008-0227-cf64-30995519`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 7300, 'eta': '5h43m', 'grad': 0.1245, 'ce': 2.668}.
- w2: `mose-width-w2-p1-1x8-1008-0227-3754-36807631`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 7380, 'eta': '5h35m', 'grad': 0.1216, 'ce': 2.584}.
- g1: `mose-width-g1-p1-1x8-1008-0227-1760-42442674`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 6860, 'eta': '6h28m', 'grad': 0.024, 'ce': 2.609}.

## 2026-10-08 15:10 HKT — monitoring

- w1: `mose-width-w1-p1-1x8-1008-0226-37ef-94920163`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 7940, 'eta': '5h10m', 'grad': 0.0227, 'ce': 2.619}.
- w1qk: `mose-width-w1qk-p1-1x8-1008-0227-b83b-25485220`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 7930, 'eta': '5h11m', 'grad': 0.0169, 'ce': 2.627}.
- w1m: `mose-width-w1m-p1-1x8-1008-0227-cf64-30995519`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 7930, 'eta': '5h11m', 'grad': 0.0843, 'ce': 2.644}.
- w2: `mose-width-w2-p1-1x8-1008-0227-3754-36807631`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 7990, 'eta': '5h5m', 'grad': 0.0555, 'ce': 2.615}.
- g1: `mose-width-g1-p1-1x8-1008-0227-1760-42442674`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 7440, 'eta': '5h57m', 'grad': 0.0216, 'ce': 2.609}.

## 2026-10-08 15:40 HKT — monitoring

- w1: `mose-width-w1-p1-1x8-1008-0226-37ef-94920163`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 8530, 'eta': '4h42m', 'grad': 0.1187, 'ce': 2.62}.
- w1qk: `mose-width-w1qk-p1-1x8-1008-0227-b83b-25485220`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 8520, 'eta': '4h43m', 'grad': 0.0171, 'ce': 2.608}.
- w1m: `mose-width-w1m-p1-1x8-1008-0227-cf64-30995519`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 8520, 'eta': '4h43m', 'grad': 0.0494, 'ce': 2.594}.
- w2: `mose-width-w2-p1-1x8-1008-0227-3754-36807631`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 8610, 'eta': '4h36m', 'grad': 0.0851, 'ce': 2.669}.
- g1: `mose-width-g1-p1-1x8-1008-0227-1760-42442674`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 8000, 'eta': '5h29m', 'grad': 0.0289, 'ce': 2.586}.

## 2026-10-08 16:10 HKT — monitoring

- w1: `mose-width-w1-p1-1x8-1008-0226-37ef-94920163`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 9150, 'eta': '4h12m', 'grad': 0.179, 'ce': 2.64}.
- w1qk: `mose-width-w1qk-p1-1x8-1008-0227-b83b-25485220`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 9150, 'eta': '4h12m', 'grad': 0.0267, 'ce': 2.639}.
- w1m: `mose-width-w1m-p1-1x8-1008-0227-cf64-30995519`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 9140, 'eta': '4h13m', 'grad': 0.1236, 'ce': 2.696}.
- w2: `mose-width-w2-p1-1x8-1008-0227-3754-36807631`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 9230, 'eta': '4h6m', 'grad': 0.0921, 'ce': 2.611}.
- g1: `mose-width-g1-p1-1x8-1008-0227-1760-42442674`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 8580, 'eta': '4h59m', 'grad': 0.0264, 'ce': 2.647}.

## 2026-10-08 16:40 HKT — monitoring

- w1: `mose-width-w1-p1-1x8-1008-0226-37ef-94920163`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 9780, 'eta': '3h41m', 'grad': 0.0145, 'ce': 2.564}.
- w1qk: `mose-width-w1qk-p1-1x8-1008-0227-b83b-25485220`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 9770, 'eta': '3h42m', 'grad': 0.0261, 'ce': 2.638}.
- w1m: `mose-width-w1m-p1-1x8-1008-0227-cf64-30995519`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 9760, 'eta': '3h42m', 'grad': 0.0193, 'ce': 2.673}.
- w2: `mose-width-w2-p1-1x8-1008-0227-3754-36807631`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 9860, 'eta': '3h35m', 'grad': 0.037, 'ce': 2.597}.
- g1: `mose-width-g1-p1-1x8-1008-0227-1760-42442674`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 9170, 'eta': '4h28m', 'grad': 0.0216, 'ce': 2.606}.

## 2026-10-08 18:41 HKT — monitoring

- w1: `mose-width-w1-p1-1x8-1008-0226-37ef-94920163`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 12250, 'eta': '1h41m', 'grad': 0.0311, 'ce': 2.58}.
- w1qk: `mose-width-w1qk-p1-1x8-1008-0227-b83b-25485220`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 12240, 'eta': '1h41m', 'grad': 0.0141, 'ce': 2.58}.
- w1m: `mose-width-w1m-p1-1x8-1008-0227-cf64-30995519`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 12240, 'eta': '1h41m', 'grad': 0.0136, 'ce': 2.595}.
- w2: `mose-width-w2-p1-1x8-1008-0227-3754-36807631`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 12350, 'eta': '1h35m', 'grad': 0.0171, 'ce': 2.531}.
- g1: `mose-width-g1-p1-1x8-1008-0227-1760-42442674`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 11490, 'eta': '2h27m', 'grad': 0.0246, 'ce': 2.583}.

## 2026-10-08 20:21 HKT — scheduler snapshot and W2 completion

- `mose-width-w1-p1-1x8-1008-0226-37ef-94920163`: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}.
- `mose-width-w1qk-p1-1x8-1008-0227-b83b-25485220`: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}.
- `mose-width-w1m-p1-1x8-1008-0227-cf64-30995519`: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}.
- `mose-width-w2-p1-1x8-1008-0227-3754-36807631`: Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}.
- `mose-width-g1-p1-1x8-1008-0227-1760-42442674`: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}.

W2最终step14306于20:17:12保存成功，20:17:15 Training complete，调度器Succeeded / active=0 / succeeded=1。[完成证据](completion_w2.log)。W1、W1QK、W1M、G1仍运行。新增Dense FFN三个任务均为arch pool 1×8低优，Inqueue；两批共7条未完成。[当前监控名单](../Dense_FFN_Linear_RMS/RUN_STATUS.md)。

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

## 2026-10-09 00:53 HKT — 五条完整末期Train CE汇总；停止额外评测

完整七条W&B历史均覆盖steps1–14306，无缺失/重复。统一末期窗口13307–14306，各1000个有效点，排名使用原始未平滑均值。C6异常12516在窗口外，无影响。

| 配置 | 末1000步Train CE | 相对C6 |
| --- | ---: | ---: |
| G1 | 2.473732 | −0.001031 |
| C6 | 2.474763 | 0 |
| C2 | 2.475668 | +0.000905 |
| W1 | 2.481117 | +0.006354 |
| W1-QK | 2.481223 | +0.006460 |
| W2 | 2.490827 | +0.016064 |
| W1-M | 2.497229 | +0.022466 |

缩宽组合未超过C2/C6；回补瓶颈使W1比W1-M改善0.016112，仍不足以补偿缩宽损失。W1-QK比W1高0.000106。G1增加32参数、单次末期改善0.001031，作为小幅改善候选，尚未重复核实。W1-M少约6.8%参数，单列。

用户最新要求“不需要评测，只要分析一下末期的train ce即可”。新增分析/评测job `xtuner-width-gate-analysis-r1-1009-0040-10561557` 于00:50:27收到stop确认，00:53:00核实Stopped/active=0/stopped=1。此前已提取完整训练历史，只有W1完成了额外验证；部分验证文件保留，但不纳入本报告。不再继续/重提评测，也没有停止其他训练或部署。

SSH网关的W1精确 `.wandb` 路径实查可读；后续读取记录可通过SSH传到本机分析，不必单为读取日志申请GPU。本次评测job和终态记录见[analysis_jobs.json](analysis_jobs.json)。

[末期结果与完整W&B名称](COMPLETED_RESULTS_20261009.md) / [末期比较JSON](TAIL_TRAIN_CE_COMPARISON.json) / [完整历史及来源哈希](analysis/completed_20261009/FULL_TRAIN_CE.json)。
