# Dense sharing runs

2026-10-05 13:01 HKT. User authorized S1/S2/S3, dimensions round down, monitor every30min, prefer16GPUs. Previous no-arch-queue preference retained.

Local tests:3 passed, actual module/full-model counts, shared KV collapsed-dense equivalence, shared UG formula and backward. Bash syntax passed.

Attempt r1 arch2x8 priority1:
- S1 mose-dense-sharing-s1-r1-p1-2x8-1005-35757879
- S2 mose-dense-sharing-s2-r1-p1-2x8-1005-50412346
- S3 mose-dense-sharing-s3-r1-p1-2x8-1005-63659020
All Inqueue with unassigned nodes12:59:41; stopped12:59:57 and verified active0/stopped2 at12:59:58. None trained, no W&B segment to sync.

Pool r2 actual submissions (each1x8, priority1, no auto restart):
- S1: `mose-dense-sharing-s1-r2-p1-1x8-pool-1005-13108302`
- S2: `mose-dense-sharing-s2-r2-p1-1x8-pool-1005-26282060`
- S3: `mose-dense-sharing-s3-r2-p1-1x8-pool-1005-42104108`

## 2026-10-05 13:01 HKT — monitoring

- S1: Inqueue; {'active': 0, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics None.
- S2: Inqueue; {'active': 0, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics None.
- S3: Inqueue; {'active': 0, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics None.

## 2026-10-05 13:08 HKT — monitoring

- S1: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 40, 'eta': '15h15m', 'grad': 0.7875, 'ce': 7.966}.
- S2: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 10, 'eta': '14h58m', 'grad': 2.619, 'ce': 11.61}.
- S3: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics None.

## 2026-10-05 13:08 HKT — All pool jobs allocated

S1 training40 steps; S2 training10; S3 distributed model initialized, no training metric yet. All active1/failed0, mesh dp_replicate1/dp_shard8. User authorizes arch16/32 when resources available, but already-running jobs stay in pool.
S1 exact W&B segments: /mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261005_130419-olmo3-1b-dense-sharing-s1-linear-fp8-30b
S2 exact W&B segments: /mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261005_130632-olmo3-1b-dense-sharing-s2-linear-fp8-30b

## 2026-10-05 13:39 HKT — monitoring

- S1: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 500, 'eta': '15h32m', 'grad': 0.2092, 'ce': 3.417}.
- S2: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 490, 'eta': '14h36m', 'grad': 0.1388, 'ce': 3.389}.
- S3: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 460, 'eta': '14h55m', 'grad': 0.2111, 'ce': 3.505}.

## 2026-10-05 14:14 HKT — monitoring

- S1: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 1030, 'eta': '14h52m', 'grad': 0.0998, 'ce': 3.1}.
- S2: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 1010, 'eta': '14h43m', 'grad': 0.1256, 'ce': 3.031}.
- S3: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 990, 'eta': '14h38m', 'grad': 0.18, 'ce': 3.119}.

## 2026-10-05 14:46 HKT — monitoring

- S1: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 1500, 'eta': '14h19m', 'grad': 0.1064, 'ce': 2.923}.
- S2: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 1490, 'eta': '13h59m', 'grad': 0.0808, 'ce': 2.878}.
- S3: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 1460, 'eta': '14h12m', 'grad': 0.1293, 'ce': 2.974}.
- Arch 4x8 prediction: ['📌 任务调度核心状态：❌ 任务不可调度', '  - 总副本数量：4', '  - 可调度数量：0', '  - 不可调度数量：4'].
- Arch 2x8 prediction: ['📌 任务调度核心状态：❌ 任务不可调度', '  - 总副本数量：2', '  - 可调度数量：0', '  - 不可调度数量：2'].

## 2026-10-05 15:16 HKT — monitoring

- S1: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 1980, 'eta': '13h36m', 'grad': 0.077, 'ce': 2.844}.
- S2: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 1960, 'eta': '13h31m', 'grad': 0.0476, 'ce': 2.893}.
- S3: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 1910, 'eta': '13h44m', 'grad': 0.0851, 'ce': 2.869}.
- Arch 4x8 prediction: ['📌 任务调度核心状态：❌ 任务不可调度', '  - 总副本数量：4', '  - 可调度数量：0', '  - 不可调度数量：4'].
- Arch 2x8 prediction: ['📌 任务调度核心状态：❌ 任务不可调度', '  - 总副本数量：2', '  - 可调度数量：0', '  - 不可调度数量：2'].

## 2026-10-05 15:47 HKT — monitoring

- S1: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 2440, 'eta': '13h7m', 'grad': 0.0458, 'ce': 2.734}.
- S2: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 2420, 'eta': '13h2m', 'grad': 0.0239, 'ce': 2.782}.
- S3: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 2370, 'eta': '13h14m', 'grad': 0.0321, 'ce': 2.811}.
- Arch 4x8 prediction: ['📌 任务调度核心状态：❌ 任务不可调度', '  - 总副本数量：4', '  - 可调度数量：0', '  - 不可调度数量：4'].
- Arch 2x8 prediction: ['📌 任务调度核心状态：❌ 任务不可调度', '  - 总副本数量：2', '  - 可调度数量：0', '  - 不可调度数量：2'].

## 2026-10-05 16:17 HKT — monitoring

- S1: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 2900, 'eta': '12h37m', 'grad': 0.0272, 'ce': 2.714}.
- S2: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 2880, 'eta': '12h33m', 'grad': 0.0259, 'ce': 2.761}.
- S3: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 2830, 'eta': '12h44m', 'grad': 0.0239, 'ce': 2.78}.
- Arch 4x8 prediction: ['📌 任务调度核心状态：❌ 任务不可调度', '  - 总副本数量：4', '  - 可调度数量：0', '  - 不可调度数量：4'].
- Arch 2x8 prediction: ['📌 任务调度核心状态：❌ 任务不可调度', '  - 总副本数量：2', '  - 可调度数量：0', '  - 不可调度数量：2'].

## 2026-10-05 16:48 HKT — monitoring

- S1: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 3360, 'eta': '12h7m', 'grad': 0.0346, 'ce': 2.654}.
- S2: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 3350, 'eta': '12h2m', 'grad': 0.0473, 'ce': 2.695}.
- S3: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 3290, 'eta': '12h13m', 'grad': 0.0202, 'ce': 2.683}.
- Arch 4x8 prediction: ['📌 任务调度核心状态：❌ 任务不可调度', '  - 总副本数量：4', '  - 可调度数量：0', '  - 不可调度数量：4'].
- Arch 2x8 prediction: ['📌 任务调度核心状态：❌ 任务不可调度', '  - 总副本数量：2', '  - 可调度数量：0', '  - 不可调度数量：2'].

## 2026-10-05 17:18 HKT — monitoring

- S1: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 3810, 'eta': '11h37m', 'grad': 0.034, 'ce': 2.63}.
- S2: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 3810, 'eta': '11h32m', 'grad': 0.0189, 'ce': 2.616}.
- S3: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 3750, 'eta': '11h43m', 'grad': 0.0315, 'ce': 2.605}.
- Arch 4x8 prediction: ['📌 任务调度核心状态：❌ 任务不可调度', '  - 总副本数量：4', '  - 可调度数量：0', '  - 不可调度数量：4'].
- Arch 2x8 prediction: ['📌 任务调度核心状态：❌ 任务不可调度', '  - 总副本数量：2', '  - 可调度数量：0', '  - 不可调度数量：2'].

## 2026-10-06 10:42 HKT — monitoring

- S1: Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 14300, 'eta': '1m', 'grad': 0.0088, 'ce': 2.39}.
- S2: Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 14300, 'eta': '1m', 'grad': 0.0081, 'ce': 2.381}.
- S3: Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 14300, 'eta': '1m', 'grad': 0.0073, 'ce': 2.392}.
- Arch 4x8 prediction: ['📌 任务调度核心状态：✅ 任务可调度', '  - 总副本数量：4', '  - 可调度数量：4', '  - 不可调度数量：0'].
- Arch 2x8 prediction: ['📌 任务调度核心状态：✅ 任务可调度', '  - 总副本数量：2', '  - 可调度数量：2', '  - 不可调度数量：0'].

## 2026-10-06 — Completion audit

All three original pool jobs Succeeded active0/succeeded1; no duplicate running attempts. Completed14306 and final-checkpoint saves verified in scheduler logs. Full offline histories14306/14306 valid CE rows for all3 and dense reference, no exclusions. Results in RESULTS.md and train_ce_comparison.json. S1/S2 exact W&B segments Finished syncing verified; S3 sync launched, awaiting completion. arch16/32 now schedulable but no remaining queued work requires submission.

S3 Finished syncing verified. All3 runs completed and synchronized; final Train CE comparison persisted. No active jobs or pending uploads remain; monitoring objective complete.
