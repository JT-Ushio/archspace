# UG partial-sharing runs

2026-10-06: user requests continuing experiments, arch16/32 when available,
and allows many concurrent pool1x8 experiments. Low priority1; 30-minute monitoring.

Prepared five independent configurations. P0/P2/P3/P4 exactly match the
797,050,880-parameter reference. P1=794,429,440 (-0.329%): shared rank rounded down.
All other ranks/activations fixed: Q/K/V/O512 Identity, Down512 RMSNorm,
native SwiGLU retained. Seeds12536/34521, standard30B recipe.

2026-10-06 11:14 HKT: authoritative arch predictions4x8 and2x8 schedulable.
Old S1/S2/S3 jobs rechecked Succeeded, active0/succeeded1. No resubmission.

Local formula/gradient/count validation in progress. No new GPU job submitted yet.

Submitted P2: `mose-ug-p2-r1-p1-4x8-arch-1006-1123-f1d4-96742035`, arch 4x8, priority1; script `/mnt/shared-storage-gpfs2/intern-pretrain-shared02/shijiayang-p/archspace-MoSE/reproduce/olmo-core-backend/scripts/train/UG_Partial_Sharing/run_p2.sh`.

Submitted P3: `mose-ug-p3-r1-p1-2x8-arch-1006-1123-5b87-34699678`, arch 2x8, priority1; script `/mnt/shared-storage-gpfs2/intern-pretrain-shared02/shijiayang-p/archspace-MoSE/reproduce/olmo-core-backend/scripts/train/UG_Partial_Sharing/run_p3.sh`.

Submitted P0: `mose-ug-p0-r1-p1-1x8-pool-1006-1123-7104-77485485`, pool 1x8, priority1; script `/mnt/shared-storage-gpfs2/intern-pretrain-shared02/shijiayang-p/archspace-MoSE/reproduce/olmo-core-backend/scripts/train/UG_Partial_Sharing/run_p0.sh`.

Submitted P1: `mose-ug-p1-r1-p1-1x8-pool-1006-1123-6d50-83419854`, pool 1x8, priority1; script `/mnt/shared-storage-gpfs2/intern-pretrain-shared02/shijiayang-p/archspace-MoSE/reproduce/olmo-core-backend/scripts/train/UG_Partial_Sharing/run_p1.sh`.

Submitted P4: `mose-ug-p4-r1-p1-1x8-pool-1006-1123-1580-89006511`, pool 1x8, priority1; script `/mnt/shared-storage-gpfs2/intern-pretrain-shared02/shijiayang-p/archspace-MoSE/reproduce/olmo-core-backend/scripts/train/UG_Partial_Sharing/run_p4.sh`.

Local checks passed:154 tests including exact formula/gradient, module/full-model counts, and legacy sharing compatibility. All5 full configs validated; global batch divisible for8/16/32GPU. Shell/Python syntax passed.

Pool command alias resolved from the gateway rejection: gpu_run pool is unsupported and failed before invoking rjob. The authorized pool jobs were then submitted through gpu_run archp; charged group llmarchitecture_gpu_pool and namespace ailab-llmarchitecture confirmed in submission responses. No duplicate job was created.

All5 unique job IDs confirmed; auto restart disabled. P2 already Running active4/failed0 at11:25:48 HKT. Initial verification pending for the remaining jobs. Monitoring uses1800-second intervals.

## 2026-10-06 11:27 HKT — monitoring

- p2: `mose-ug-p2-r1-p1-4x8-arch-1006-1123-f1d4-96742035`; Running; {'active': 4, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
- p3: `mose-ug-p3-r1-p1-2x8-arch-1006-1123-5b87-34699678`; Inqueue; {'active': 0, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
- p0: `mose-ug-p0-r1-p1-1x8-pool-1006-1123-7104-77485485`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
- p1: `mose-ug-p1-r1-p1-1x8-pool-1006-1123-6d50-83419854`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
- p4: `mose-ug-p4-r1-p1-1x8-pool-1006-1123-1580-89006511`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.

11:28 HKT: P3 arch2x8 remained Inqueue with unassigned replicas on two authoritative checks. Per user no-arch-queue preference, stopped old job; confirmed Stopped/active0/stopped2. No assigned node or training/W&B segment, so no sync required. Preparing pool1x8 replacement with identical run/checkpoint/config.

Submitted P3: `mose-ug-p3-r2-p1-1x8-pool-1006-1128-e927-30519513`, pool 1x8, priority1; script `/mnt/shared-storage-gpfs2/intern-pretrain-shared02/shijiayang-p/archspace-MoSE/reproduce/olmo-core-backend/scripts/train/UG_Partial_Sharing/run_p3.sh`.

11:29 HKT initial startup verification: all5 jobs Running; P2 active4, each pool job active1; no failed replicas. P2 mesh(4,8), CE at step30=8.871, grad1.566; pool P0/P1/P4 mesh(1,8), initial W&B directories resolved; P3 pool mesh(1,8) initialized. Early metrics are not a quality comparison.
- p2 W&B: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_112756-olmo3-1b-ug-partial-p2-s480-p80-linear-down-rms-fp8-30b']
- p0 W&B: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_112904-olmo3-1b-ug-partial-p0-s0-p512-linear-down-rms-fp8-30b']
- p1 W&B: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_112914-olmo3-1b-ug-partial-p1-s560-p0-linear-down-rms-fp8-30b']
- p4 W&B: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_112920-olmo3-1b-ug-partial-p4-s160-p368-linear-down-rms-fp8-30b']

Unified monitoring covers5 UG-sharing and3 FFN-reallocation jobs. Replaced only the local monitoring process to load the expanded registry; all GPU jobs remain running. Next regular check11:57 HKT, then every1800 seconds.

Generalized offline analysis verified against completed Dense/S1/S2/S3 histories: all prior full/tail/final CE reproduce to1e-12. Supports common-step anomaly sensitivity and identical11-source validation.

Submission audit: all active variants have unique scheduler IDs, config launchers and run paths; allpool accounting confirmed llmarchitecture_gpu_pool. No duplicate live attempts. Code remains on arch/MoSE-no-kvcache.

## 2026-10-06 11:57 HKT — monitoring

- p2: `mose-ug-p2-r1-p1-4x8-arch-1006-1123-f1d4-96742035`; Running; {'active': 4, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 2000, 'eta': '2h52m', 'grad': 0.1234, 'ce': 3.003}.
- p0: `mose-ug-p0-r1-p1-1x8-pool-1006-1123-7104-77485485`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 540, 'eta': '11h32m', 'grad': 0.1506, 'ce': 3.47}.
- p1: `mose-ug-p1-r1-p1-1x8-pool-1006-1123-6d50-83419854`; Failed; {'active': 0, 'succeeded': 0, 'failed': 1, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
- p4: `mose-ug-p4-r1-p1-1x8-pool-1006-1123-1580-89006511`; Failed; {'active': 0, 'succeeded': 0, 'failed': 1, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 490, 'eta': '12h2m', 'grad': 0.1962, 'ce': 3.527}.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_112920-olmo3-1b-ug-partial-p4-s160-p368-linear-down-rms-fp8-30b', '/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_112920-olmo3-1b-ug-partial-p4-s160-p368-linear-down-rms-fp8-30b/logs'].
- p3: `mose-ug-p3-r2-p1-1x8-pool-1006-1128-e927-30519513`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 470, 'eta': '11h59m', 'grad': 0.198, 'ce': 3.534}.
- f1: `mose-ffn-f1-r1-p1-1x8-pool-1006-1135-4cae-99748743`; Failed; {'active': 0, 'succeeded': 0, 'failed': 1, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
- f2: `mose-ffn-f2-r1-p1-1x8-pool-1006-1135-6c39-5628140`; Failed; {'active': 0, 'succeeded': 0, 'failed': 1, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_114106-olmo3-1b-ffn-budget-f2-ug256-down1024-unshared-linear-rms-fp8-30b'].
- f3: `mose-ffn-f3-r1-p1-1x8-pool-1006-1135-0526-12030032`; Failed; {'active': 0, 'succeeded': 0, 'failed': 1, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_114123-olmo3-1b-ffn-budget-f3-ug480-down672-shared-linear-rms-fp8-30b'].

Recovery p1: old `mose-ug-p1-r1-p1-1x8-pool-1006-1123-6d50-83419854` verified Failed/active0. Exact old W&B segment Finished syncing. Same run/checkpoint/config; only disposable compiler/temp caches redirected to job-local /tmp. New pool1x8 priority1 attempt follows.

## 2026-10-06 11:57–12:05 HKT — failure diagnosis and correction

P1/P4/F1/F3 have explicit Disk quota exceeded in GCC/Inductor writes. F3 reports /mnt/shared-storage-user/jitao/tmp_data/cc*.s. F2 reports CUDA driver error: device kernel image is invalid; shared-cache corruption is a hypothesis, not a verified root cause. SIGTERM lines are the distributed shutdown after the primary failures.
Launchers now create fresh per-job/per-node disposable TMPDIR, TorchInductor, Triton and CUDA caches under /tmp before invoking hceph/Python. Setup and bash syntax tested locally. Work/data/checkpoint paths and all architecture/hyperparameters are preserved. Only verified Failed/active0 attempts are synchronized and replaced; healthy jobs continue.
Arch restart prediction:4x8 and2x8 unavailable (only1 replica schedulable); authorized replacements use pool1x8 priority1.

P1 replacement actual ID `mose-recover-p1-r2-p1-1x8-pool-1006-1204-efa-886b9` confirmed from explicit created response and authoritative rjob get. Scheduler normalized the requested name; updated parser, no second submission.

Recovery p4: old `mose-ug-p4-r1-p1-1x8-pool-1006-1123-1580-89006511` verified Failed/active0. Exact old W&B segment Finished syncing. Same run/checkpoint/config; only disposable compiler/temp caches redirected to job-local /tmp. New pool1x8 priority1 attempt follows.

Submitted P4: `mose-recover-p4-r2-p1-1x8-pool-1006-1207-0d3-44098`, pool 1x8, priority1; script `/mnt/shared-storage-gpfs2/intern-pretrain-shared02/shijiayang-p/archspace-MoSE/reproduce/olmo-core-backend/scripts/train/UG_Partial_Sharing/run_p4.sh`.

12:14 HKT recovery verification: P1/P4/F1 resumed step0, passed dry-run and emitted finite train CE. F2/F3 replacements Running active1 on pool1x8, fresh job-local caches and (dp_replicate1,dp_shard8) verified; first CE pending at this check. Old F2 upload encountered transaction-log unexpected EOF; original record preserved, completion is not claimed. Same hyperparameters and outputs retained.

## 2026-10-06 12:27 HKT — monitoring

- p2: `mose-ug-p2-r1-p1-4x8-arch-1006-1123-f1d4-96742035`; Running; {'active': 4, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 4220, 'eta': '2h20m', 'grad': 0.0929, 'ce': 2.765}.
- p0: `mose-ug-p0-r1-p1-1x8-pool-1006-1123-7104-77485485`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 1150, 'eta': '10h59m', 'grad': 0.1282, 'ce': 3.155}.
- p3: `mose-ug-p3-r2-p1-1x8-pool-1006-1128-e927-30519513`; Failed; {'active': 0, 'succeeded': 0, 'failed': 1, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_113123-olmo3-1b-ug-partial-p3-s320-p224-linear-down-rms-fp8-30b', '/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_113123-olmo3-1b-ug-partial-p3-s320-p224-linear-down-rms-fp8-30b/logs'].
- p1: `mose-recover-p1-r2-p1-1x8-pool-1006-1204-efa-886b9`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 390, 'eta': '10h58m', 'grad': 0.208, 'ce': 3.728}.
- p4: `mose-recover-p4-r2-p1-1x8-pool-1006-1207-0d3-44098`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 310, 'eta': '12h9m', 'grad': 0.309, 'ce': 4.061}.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_120940-olmo3-1b-ug-partial-p4-s160-p368-linear-down-rms-fp8-30b'].
- f1: `mose-recover-f1-r2-p1-1x8-pool-1006-1207-08d-af98d`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 320, 'eta': '11h21m', 'grad': 0.2059, 'ce': 3.966}.
- f2: `mose-recover-f2-r2-p1-1x8-pool-1006-1213-0e1-bdefa`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 210, 'eta': '11h23m', 'grad': 0.3804, 'ce': 5.104}.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_121541-olmo3-1b-ffn-budget-f2-ug256-down1024-unshared-linear-rms-fp8-30b'].
- f3: `mose-recover-f3-r2-p1-1x8-pool-1006-1213-ccc-5d862`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 200, 'eta': '11h16m', 'grad': 0.4418, 'ce': 5.161}.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_121614-olmo3-1b-ffn-budget-f3-ug480-down672-shared-linear-rms-fp8-30b'].

Recovery p3: old `mose-ug-p3-r2-p1-1x8-pool-1006-1128-e927-30519513` verified Failed/active0. Exact old W&B synchronization state: started. Same run/checkpoint/config; only disposable compiler/temp caches redirected to job-local /tmp. New pool1x8 priority1 attempt follows.

Submitted P3: `mose-recover-p3-r3-p1-1x8-pool-1006-1228-c38-15484`, pool 1x8, priority1; script `/mnt/shared-storage-gpfs2/intern-pretrain-shared02/shijiayang-p/archspace-MoSE/reproduce/olmo-core-backend/scripts/train/UG_Partial_Sharing/run_p3.sh`.

12:27–12:30 HKT: P2 arch32 step4220 CE2.765; P0 pool8 step1150 CE3.155. P1/P4/F1/F2/F3 pool8 recoveries all Running with finite CE. P3 original pool attempt failed at first LM evaluation with Inductor Errno122; verified Failed/active0, exact logs submitted for synchronization before pool8 r3 replacement. Arch2x8/4x8 prediction has only1 schedulable replica; no arch job queued.

## 2026-10-06 12:57 HKT — monitoring

- p2: `mose-ug-p2-r1-p1-4x8-arch-1006-1123-f1d4-96742035`; Running; {'active': 4, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 6450, 'eta': '1h48m', 'grad': 0.0162, 'ce': 2.711}.
- p0: `mose-ug-p0-r1-p1-1x8-pool-1006-1123-7104-77485485`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 1760, 'eta': '10h27m', 'grad': 0.135, 'ce': 2.966}.
- p1: `mose-recover-p1-r2-p1-1x8-pool-1006-1204-efa-886b9`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 990, 'eta': '10h47m', 'grad': 0.18, 'ce': 3.221}.
- p4: `mose-recover-p4-r2-p1-1x8-pool-1006-1207-0d3-44098`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 860, 'eta': '12h6m', 'grad': 0.1717, 'ce': 3.217}.
- p3: `mose-recover-p3-r3-p1-1x8-pool-1006-1228-c38-15484`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 470, 'eta': '11h49m', 'grad': 0.1872, 'ce': 3.537}.
- f1: `mose-recover-f1-r2-p1-1x8-pool-1006-1207-08d-af98d`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 920, 'eta': '11h11m', 'grad': 0.2571, 'ce': 3.193}.
- f2: `mose-recover-f2-r2-p1-1x8-pool-1006-1213-0e1-bdefa`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 810, 'eta': '11h17m', 'grad': 0.2645, 'ce': 3.278}.
- f3: `mose-recover-f3-r2-p1-1x8-pool-1006-1213-ccc-5d862`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 810, 'eta': '11h9m', 'grad': 0.2022, 'ce': 3.259}.

12:57 HKT: all8 Running with authoritative active replica counts (P2 arch4, allother pool1). P1/P4/F1/F2/F3 have advanced beyond eval500 after moving disposable caches to node-local /tmp; no recurrence in the current logs. P3 step470, its eval500 still pending at this poll. [{"nodes": 4, "summary": ["📌 任务调度核心状态：❌ 任务不可调度", "  - 总副本数量：4", "  - 可调度数量：1", "  - 不可调度数量：3"]}, {"nodes": 2, "summary": ["📌 任务调度核心状态：❌ 任务不可调度", "  - 总副本数量：2", "  - 可调度数量：1", "  - 不可调度数量：1"]}]

## 2026-10-06 13:28 HKT — monitoring

- p2: `mose-ug-p2-r1-p1-4x8-arch-1006-1123-f1d4-96742035`; Running; {'active': 4, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 8650, 'eta': '1h18m', 'grad': 0.036, 'ce': 2.622}.
- p0: `mose-ug-p0-r1-p1-1x8-pool-1006-1123-7104-77485485`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 2370, 'eta': '9h55m', 'grad': 0.0604, 'ce': 2.913}.
- p1: `mose-recover-p1-r2-p1-1x8-pool-1006-1204-efa-886b9`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 1610, 'eta': '10h28m', 'grad': 0.187, 'ce': 3.022}.
- p4: `mose-recover-p4-r2-p1-1x8-pool-1006-1207-0d3-44098`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 1420, 'eta': '11h34m', 'grad': 0.1528, 'ce': 3.11}.
- p3: `mose-recover-p3-r3-p1-1x8-pool-1006-1228-c38-15484`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 1020, 'eta': '11h55m', 'grad': 0.2457, 'ce': 3.193}.
- f1: `mose-recover-f1-r2-p1-1x8-pool-1006-1207-08d-af98d`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 1500, 'eta': '10h54m', 'grad': 0.194, 'ce': 3.051}.
- f2: `mose-recover-f2-r2-p1-1x8-pool-1006-1213-0e1-bdefa`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 1410, 'eta': '10h48m', 'grad': 0.1685, 'ce': 3.077}.
- f3: `mose-recover-f3-r2-p1-1x8-pool-1006-1213-ccc-5d862`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 1420, 'eta': '10h39m', 'grad': 0.1518, 'ce': 3.103}.

13:28 HKT: all8 Running/active replicas. P3 recovered attempt has passed step1000 (first eval500 failure no longer recurs). Arch resource predictions: [{"nodes": 4, "summary": ["📌 任务调度核心状态：❌ 任务不可调度", "  - 总副本数量：4", "  - 可调度数量：2", "  - 不可调度数量：2"]}, {"nodes": 2, "summary": ["📌 任务调度核心状态：✅ 任务可调度", "  - 总副本数量：2", "  - 可调度数量：2", "  - 不可调度数量：0"]}]

13:34–13:50 local preparation: repository writes also hit Errno122. Cleared only disposable Python bytecode for this session's experiment/test modules and modified feed_forward; source, data and checkpoints preserved. Writes then succeeded. New P0qk/P2qk launchers disable bytecode writes and reuse fresh job-local compiler caches. Both full configs resolve exact797,050,880params/seeds12536,34521/30B WSD2000+20% FP8 recipe; unit validation pending before submission.

13:47 HKT: QK interaction validation4tests passed. Both exact rank512-budget configs preserve all model/recipe fields except QK intermediate SiLU and tracking group; actual meta FFN/attention counts checked, batches divisible at8/16/32. Fresh arch4x8 prediction4/4 schedulable; P2qk usesarch32, P0qkpool8, allpriority1. Submission responses follow.

Submitted P2QK: `mose-ug-p2qk-r1-p1-4x8-arch-1006-1347-8485-7-afb90`, arch 4x8, priority1; script `/mnt/shared-storage-gpfs2/intern-pretrain-shared02/shijiayang-p/archspace-MoSE/reproduce/olmo-core-backend/scripts/train/UG_Partial_Sharing/run_p2qk.sh`.

## 2026-10-06 13:53 HKT — monitoring

- p2: `mose-ug-p2-r1-p1-4x8-arch-1006-1123-f1d4-96742035`; Failed; {'active': 0, 'succeeded': 0, 'failed': 4, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 8940, 'eta': '1h14m', 'grad': 0.0219, 'ce': 2.618}.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_112756-olmo3-1b-ug-partial-p2-s480-p80-linear-down-rms-fp8-30b', '/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_112756-olmo3-1b-ug-partial-p2-s480-p80-linear-down-rms-fp8-30b/logs'].
- p0: `mose-ug-p0-r1-p1-1x8-pool-1006-1123-7104-77485485`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 2870, 'eta': '9h30m', 'grad': 0.0513, 'ce': 2.814}.
- p1: `mose-recover-p1-r2-p1-1x8-pool-1006-1204-efa-886b9`; Failed; {'active': 0, 'succeeded': 0, 'failed': 1, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 1780, 'eta': '10h16m', 'grad': 0.1179, 'ce': 2.952}.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_120705-olmo3-1b-ug-partial-p1-s560-p0-linear-down-rms-fp8-30b', '/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_120705-olmo3-1b-ug-partial-p1-s560-p0-linear-down-rms-fp8-30b/logs'].
- p4: `mose-recover-p4-r2-p1-1x8-pool-1006-1207-0d3-44098`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 1870, 'eta': '11h13m', 'grad': 0.1523, 'ce': 2.968}.
- p3: `mose-recover-p3-r3-p1-1x8-pool-1006-1228-c38-15484`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 1490, 'eta': '11h18m', 'grad': 0.1566, 'ce': 3.004}.
- p2qk: `mose-ug-p2qk-r1-p1-4x8-arch-1006-1347-8485-7-afb90`; Running; {'active': 4, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 40, 'eta': '3h8m', 'grad': 0.8161, 'ce': 8.01}.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_135053-olmo3-1b-ug-partial-p2qk-s480-p80-linear-qk-silu-down-rms-fp8-30b'].
- f1: `mose-recover-f1-r2-p1-1x8-pool-1006-1207-08d-af98d`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 1990, 'eta': '10h21m', 'grad': 0.0919, 'ce': 2.945}.
- f2: `mose-recover-f2-r2-p1-1x8-pool-1006-1213-0e1-bdefa`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 1900, 'eta': '10h25m', 'grad': 0.0701, 'ce': 3.019}.
- f3: `mose-recover-f3-r2-p1-1x8-pool-1006-1213-ccc-5d862`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 1910, 'eta': '10h16m', 'grad': 0.1179, 'ce': 2.977}.

Recovery p2: old `mose-ug-p2-r1-p1-4x8-arch-1006-1123-f1d4-96742035` verified Failed/active0. Exact old W&B synchronization state: started. Model/run/hyperparameters preserved. Replacement script: /mnt/shared-storage-gpfs2/intern-pretrain-shared02/shijiayang-p/archspace-MoSE/reproduce/olmo-core-backend/scripts/train/UG_Partial_Sharing/run_p2_storage_resume.sh; allocation arch2x8 priority1.

Submitted P2: `mose-recover-p2-r2-p1-2x8-arch-1006-1355-000-c0361`, arch 2x8, priority1; script `/mnt/shared-storage-gpfs2/intern-pretrain-shared02/shijiayang-p/archspace-MoSE/reproduce/olmo-core-backend/scripts/train/UG_Partial_Sharing/run_p2_storage_resume.sh`.

13:55 HKT: P2 Failed/active0 at step8940; traceback is checkpointer._save_train_state into the GPFS2 step8940-tmp directory, not compiler cache. P1 also Failed/active0 at first1788 save. Old checkpoints are retained. P2 replacement explicitly requires old step7152, writes future checkpoints to GPFS1 shared-storage-ailab-llmfudan/taoji/ckpts/Olmo3-MoSE, and performs64GiB free-space/16MiB write preflight inside the GPU job. Unchanged model/data/30B recipe; arch16 priority1. Exact old W&B uploads recorded before replacements.

## 2026-10-06 13:58 HKT — monitoring

- p0: `mose-ug-p0-r1-p1-1x8-pool-1006-1123-7104-77485485`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 2990, 'eta': '9h23m', 'grad': 0.0469, 'ce': 2.827}.
- p1: `mose-recover-p1-r2-p1-1x8-pool-1006-1204-efa-886b9`; Failed; {'active': 0, 'succeeded': 0, 'failed': 1, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 1780, 'eta': '10h16m', 'grad': 0.1179, 'ce': 2.952}.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_120705-olmo3-1b-ug-partial-p1-s560-p0-linear-down-rms-fp8-30b', '/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_120705-olmo3-1b-ug-partial-p1-s560-p0-linear-down-rms-fp8-30b/logs'].
- p4: `mose-recover-p4-r2-p1-1x8-pool-1006-1207-0d3-44098`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 1980, 'eta': '11h5m', 'grad': 0.1179, 'ce': 2.967}.
- p3: `mose-recover-p3-r3-p1-1x8-pool-1006-1228-c38-15484`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 1590, 'eta': '11h21m', 'grad': 0.1204, 'ce': 3.043}.
- p2qk: `mose-ug-p2qk-r1-p1-4x8-arch-1006-1347-8485-7-afb90`; Running; {'active': 4, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 450, 'eta': '3h4m', 'grad': 0.1733, 'ce': 3.595}.
- p2: `mose-recover-p2-r2-p1-2x8-arch-1006-1355-000-c0361`; Running; {'active': 2, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_135817-olmo3-1b-ug-partial-p2-s480-p80-linear-down-rms-fp8-30b'].
- f1: `mose-recover-f1-r2-p1-1x8-pool-1006-1207-08d-af98d`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 2100, 'eta': '10h20m', 'grad': 0.0862, 'ce': 2.967}.
- f2: `mose-recover-f2-r2-p1-1x8-pool-1006-1213-0e1-bdefa`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 1990, 'eta': '10h19m', 'grad': 0.0785, 'ce': 2.959}.
- f3: `mose-recover-f3-r2-p1-1x8-pool-1006-1213-ccc-5d862`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 2010, 'eta': '10h17m', 'grad': 0.0773, 'ce': 2.958}.

Recovery p1: old `mose-recover-p1-r2-p1-1x8-pool-1006-1204-efa-886b9` verified Failed/active0. Exact old W&B synchronization state: finished. Model/run/hyperparameters preserved. Replacement script: /mnt/shared-storage-gpfs2/intern-pretrain-shared02/shijiayang-p/archspace-MoSE/reproduce/olmo-core-backend/scripts/train/UG_Partial_Sharing/run_p1_storage_resume.sh; allocation pool1x8 priority1.

Submitted P1: `mose-recover-p1-r3-p1-1x8-pool-1006-1358-5ee-ff2d5`, pool 1x8, priority1; script `/mnt/shared-storage-gpfs2/intern-pretrain-shared02/shijiayang-p/archspace-MoSE/reproduce/olmo-core-backend/scripts/train/UG_Partial_Sharing/run_p1_storage_resume.sh`.

Submitted P0QK: `mose-ug-p0qk-r1-p1-1x8-pool-1006-1347-459e-3-74da9`, pool 1x8, priority1; script `/mnt/shared-storage-gpfs2/intern-pretrain-shared02/shijiayang-p/archspace-MoSE/reproduce/olmo-core-backend/scripts/train/UG_Partial_Sharing/run_p0qk.sh`.

Current batch frozen by user request: P0–P4, F1–F3, P0qk/P2qk (10 configs). Submit/recover these, unified30min monitoring; select and launch a next batch only after the whole current batch finishes and complete Train CE comparison. P2 storage preflight on GPFS1 passed: 436GiB available,16MiB write/fsync succeeded. P1 restores original step0 into same alternate root, model/hyperparameters preserved; P0qk uses that root from initialization.

Recovery p2: old `mose-recover-p2-r2-p1-2x8-arch-1006-1355-000-c0361` verified Stopped/active0. Exact old W&B synchronization state: started. Model/run/hyperparameters preserved. Replacement script: /mnt/shared-storage-gpfs2/intern-pretrain-shared02/shijiayang-p/archspace-MoSE/reproduce/olmo-core-backend/scripts/train/UG_Partial_Sharing/run_p2_storage_resume.sh; allocation arch2x8 priority1.

Submitted P2: `mose-recover-p2-r3-p1-2x8-arch-1006-1409-7a8-a7469`, arch 2x8, priority1; script `/mnt/shared-storage-gpfs2/intern-pretrain-shared02/shijiayang-p/archspace-MoSE/reproduce/olmo-core-backend/scripts/train/UG_Partial_Sharing/run_p2_storage_resume.sh`.

14:11 HKT startup audit: P1r3 and P0qk each Running/active1, GPFS1 output preflight passed, stage1/step0 saved, finite CE through160. P1r3 actually initialized fresh with original seeds (old attempt had only stable step0); the earlier claim of a restored step0 was not verified and is corrected here. P2r2 was stopped/active0 because supplied resume overrides were lost by sourced launcher helpers; exclude that segment from quality comparisons. Both common launchers now capture positional overrides before sourcing helpers; a mock-helper argv regression passed for both families. P2r3 Running/active2 on arch16, expected strict step7152 restore into a new stage1-resume7152 save directory; runtime restore still awaiting verification. Original P2r1 W&B upload confirmed Finished syncing.

14:14 HKT: P2r3 runtime restore verified: Loading checkpoint from original GPFS2/stage1/step7152 at14:12:03, next training step7153, step7160 Train CE2.690 (console rounded). New save path is GPFS1/stage1-resume7152; arch16 mesh(2,8). Strict trainer.load_path/load_strategy overrides now reach training. Exclude old P2r2 unintended fresh-init segment; original P2r1 and stoppedr2 W&B uploads both confirmed Finished syncing. All ten batch configs are allocated and have finite training CE. Next unified poll14:28 HKT.

## 2026-10-06 14:29 HKT — monitoring

- p0: `mose-ug-p0-r1-p1-1x8-pool-1006-1123-7104-77485485`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 3580, 'eta': '8h57m', 'grad': 0.0395, 'ce': 2.746}.
- p4: `mose-recover-p4-r2-p1-1x8-pool-1006-1207-0d3-44098`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 2520, 'eta': '10h42m', 'grad': 0.0326, 'ce': 2.896}.
- p3: `mose-recover-p3-r3-p1-1x8-pool-1006-1228-c38-15484`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 2160, 'eta': '10h49m', 'grad': 0.0605, 'ce': 2.845}.
- p2qk: `mose-ug-p2qk-r1-p1-4x8-arch-1006-1347-8485-7-afb90`; Running; {'active': 4, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 2600, 'eta': '2h44m', 'grad': 0.043, 'ce': 2.824}.
- p1: `mose-recover-p1-r3-p1-1x8-pool-1006-1358-5ee-ff2d5`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 500, 'eta': '11h42m', 'grad': 0.1504, 'ce': 3.534}.
- p0qk: `mose-ug-p0qk-r1-p1-1x8-pool-1006-1347-459e-3-74da9`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 500, 'eta': '11h39m', 'grad': 0.1696, 'ce': 3.544}.
- p2: `mose-recover-p2-r3-p1-2x8-arch-1006-1409-7a8-a7469`; Running; {'active': 2, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 7700, 'eta': '3h3m', 'grad': 0.0389, 'ce': 2.623}.
- f1: `mose-recover-f1-r2-p1-1x8-pool-1006-1207-08d-af98d`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 2710, 'eta': '9h47m', 'grad': 0.0486, 'ce': 2.793}.
- f2: `mose-recover-f2-r2-p1-1x8-pool-1006-1213-0e1-bdefa`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 2600, 'eta': '9h53m', 'grad': 0.1162, 'ce': 2.845}.
- f3: `mose-recover-f3-r2-p1-1x8-pool-1006-1213-ccc-5d862`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 2620, 'eta': '9h44m', 'grad': 0.1043, 'ce': 2.891}.

## 2026-10-06 14:59 HKT — monitoring

- p0: `mose-ug-p0-r1-p1-1x8-pool-1006-1123-7104-77485485`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 4190, 'eta': '8h26m', 'grad': 0.0553, 'ce': 2.707}.
- p4: `mose-recover-p4-r2-p1-1x8-pool-1006-1207-0d3-44098`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 3090, 'eta': '10h9m', 'grad': 0.039, 'ce': 2.806}.
- p3: `mose-recover-p3-r3-p1-1x8-pool-1006-1228-c38-15484`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 2740, 'eta': '10h16m', 'grad': 0.0521, 'ce': 2.878}.
- p2qk: `mose-ug-p2qk-r1-p1-4x8-arch-1006-1347-8485-7-afb90`; Running; {'active': 4, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 4800, 'eta': '2h12m', 'grad': 0.0248, 'ce': 2.749}.
- p1: `mose-recover-p1-r3-p1-1x8-pool-1006-1358-5ee-ff2d5`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 1110, 'eta': '11h7m', 'grad': 0.188, 'ce': 3.106}.
- p0qk: `mose-ug-p0qk-r1-p1-1x8-pool-1006-1347-459e-3-74da9`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 1110, 'eta': '11h3m', 'grad': 0.1712, 'ce': 3.104}.
- p2: `mose-recover-p2-r3-p1-2x8-arch-1006-1409-7a8-a7469`; Running; {'active': 2, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 8820, 'eta': '2h31m', 'grad': 0.0258, 'ce': 2.608}.
- f1: `mose-recover-f1-r2-p1-1x8-pool-1006-1207-08d-af98d`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 3320, 'eta': '9h15m', 'grad': 0.0253, 'ce': 2.758}.
- f2: `mose-recover-f2-r2-p1-1x8-pool-1006-1213-0e1-bdefa`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 3210, 'eta': '9h21m', 'grad': 0.0471, 'ce': 2.838}.
- f3: `mose-recover-f3-r2-p1-1x8-pool-1006-1213-ccc-5d862`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 3240, 'eta': '9h12m', 'grad': 0.0514, 'ce': 2.837}.

2026-10-06 15:00 HKT arch capacity check: [{"nodes": 2, "summary": ["📌 任务调度核心状态：❌ 任务不可调度", "  - 总副本数量：2", "  - 可调度数量：0", "  - 不可调度数量：2"]}, {"nodes": 4, "summary": ["📌 任务调度核心状态：❌ 任务不可调度", "  - 总副本数量：4", "  - 可调度数量：0", "  - 不可调度数量：4"]}]. Current batch remains ten configs; no new submissions before batch completion.

## 2026-10-06 15:29 HKT — monitoring

- p0: `mose-ug-p0-r1-p1-1x8-pool-1006-1123-7104-77485485`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 4790, 'eta': '7h56m', 'grad': 0.061, 'ce': 2.699}.
- p4: `mose-recover-p4-r2-p1-1x8-pool-1006-1207-0d3-44098`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 3650, 'eta': '9h38m', 'grad': 0.0655, 'ce': 2.744}.
- p3: `mose-recover-p3-r3-p1-1x8-pool-1006-1228-c38-15484`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 3300, 'eta': '9h48m', 'grad': 0.0222, 'ce': 2.847}.
- p2qk: `mose-ug-p2qk-r1-p1-4x8-arch-1006-1347-8485-7-afb90`; Running; {'active': 4, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 6980, 'eta': '1h42m', 'grad': 0.0557, 'ce': 2.717}.
- p1: `mose-recover-p1-r3-p1-1x8-pool-1006-1358-5ee-ff2d5`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 1700, 'eta': '10h41m', 'grad': 0.1081, 'ce': 3.036}.
- p0qk: `mose-ug-p0qk-r1-p1-1x8-pool-1006-1347-459e-3-74da9`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 1720, 'eta': '10h34m', 'grad': 0.0833, 'ce': 2.942}.
- p2: `mose-recover-p2-r3-p1-2x8-arch-1006-1409-7a8-a7469`; Running; {'active': 2, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 9900, 'eta': '2h2m', 'grad': 0.0383, 'ce': 2.581}.
- f1: `mose-recover-f1-r2-p1-1x8-pool-1006-1207-08d-af98d`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 3920, 'eta': '8h45m', 'grad': 0.0185, 'ce': 2.78}.
- f2: `mose-recover-f2-r2-p1-1x8-pool-1006-1213-0e1-bdefa`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 3810, 'eta': '8h51m', 'grad': 0.0952, 'ce': 2.763}.
- f3: `mose-recover-f3-r2-p1-1x8-pool-1006-1213-ccc-5d862`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 3850, 'eta': '8h42m', 'grad': 0.0262, 'ce': 2.779}.

2026-10-06 15:30 HKT arch capacity check: [{"nodes": 2, "summary": ["📌 任务调度核心状态：❌ 任务不可调度", "  - 总副本数量：2", "  - 可调度数量：0", "  - 不可调度数量：2"]}, {"nodes": 4, "summary": ["📌 任务调度核心状态：❌ 任务不可调度", "  - 总副本数量：4", "  - 可调度数量：0", "  - 不可调度数量：4"]}]. Current batch remains ten configs; no new submissions before batch completion.

15:29 HKT checkpoint/reproducibility audit: P2 is advancing beyond the previous8940 failure point (step9900 observed) on alternateGPFS1 storage. Runtime logs warn rank RNG state was not restored because checkpoint worldsize32 differs from current16; record this resume boundary and do not describe it as bitwise-identical continuation. Log tail retrieval did not include the8940 save completion, so physical checkpoint integrity is not yet claimed. All ten jobs remain Running with expected active replicas.

15:31 HKT: authoritative P2 log with explicit --tail-lines=5000 confirms GPFS1/stage1-resume7152/step8940 saved successfully at15:02:34, after asynchronous save started15:02:29. Use positive --tail-lines for extended retrieval; negative tail previously returned only1000 lines. Checkpoint write recovery is now confirmed.

## 2026-10-06 16:00 HKT — monitoring

- p0: `mose-ug-p0-r1-p1-1x8-pool-1006-1123-7104-77485485`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 5410, 'eta': '7h25m', 'grad': 0.1387, 'ce': 2.737}.
- p4: `mose-recover-p4-r2-p1-1x8-pool-1006-1207-0d3-44098`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 4220, 'eta': '9h6m', 'grad': 0.0551, 'ce': 2.769}.
- p3: `mose-recover-p3-r3-p1-1x8-pool-1006-1228-c38-15484`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 3870, 'eta': '9h17m', 'grad': 0.0203, 'ce': 2.784}.
- p2qk: `mose-ug-p2qk-r1-p1-4x8-arch-1006-1347-8485-7-afb90`; Running; {'active': 4, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 9120, 'eta': '1h13m', 'grad': 0.0671, 'ce': 2.69}.
- p1: `mose-recover-p1-r3-p1-1x8-pool-1006-1358-5ee-ff2d5`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 2310, 'eta': '10h6m', 'grad': 0.0588, 'ce': 2.923}.
- p0qk: `mose-ug-p0qk-r1-p1-1x8-pool-1006-1347-459e-3-74da9`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 2330, 'eta': '10h1m', 'grad': 0.0796, 'ce': 2.864}.
- p2: `mose-recover-p2-r3-p1-2x8-arch-1006-1409-7a8-a7469`; Running; {'active': 2, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 10990, 'eta': '1h32m', 'grad': 0.0219, 'ce': 2.623}.
- f1: `mose-recover-f1-r2-p1-1x8-pool-1006-1207-08d-af98d`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 4500, 'eta': '8h18m', 'grad': 0.0235, 'ce': 2.703}.
- f2: `mose-recover-f2-r2-p1-1x8-pool-1006-1213-0e1-bdefa`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 4410, 'eta': '8h20m', 'grad': 0.0317, 'ce': 2.7}.
- f3: `mose-recover-f3-r2-p1-1x8-pool-1006-1213-ccc-5d862`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 4470, 'eta': '8h10m', 'grad': 0.0441, 'ce': 2.775}.

2026-10-06 16:00 HKT arch capacity check: [{"nodes": 2, "summary": ["📌 任务调度核心状态：❌ 任务不可调度", "  - 总副本数量：2", "  - 可调度数量：0", "  - 不可调度数量：2"]}, {"nodes": 4, "summary": ["📌 任务调度核心状态：❌ 任务不可调度", "  - 总副本数量：4", "  - 可调度数量：0", "  - 不可调度数量：4"]}]. Current batch remains ten configs; no new submissions before batch completion.

## 2026-10-06 18:20 HKT — monitoring

- p0: `mose-ug-p0-r1-p1-1x8-pool-1006-1123-7104-77485485`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 8220, 'eta': '5h5m', 'grad': 0.0857, 'ce': 2.638}.
- p4: `mose-recover-p4-r2-p1-1x8-pool-1006-1207-0d3-44098`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 6830, 'eta': '6h44m', 'grad': 0.024, 'ce': 2.712}.
- p3: `mose-recover-p3-r3-p1-1x8-pool-1006-1228-c38-15484`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 6500, 'eta': '6h57m', 'grad': 0.0961, 'ce': 2.633}.
- p2qk: `mose-ug-p2qk-r1-p1-4x8-arch-1006-1347-8485-7-afb90`; Succeeded; {'active': 0, 'succeeded': 4, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 14300, 'eta': '1m', 'grad': 0.0066, 'ce': 2.504}.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_135053-olmo3-1b-ug-partial-p2qk-s480-p80-linear-qk-silu-down-rms-fp8-30b', '/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_135053-olmo3-1b-ug-partial-p2qk-s480-p80-linear-qk-silu-down-rms-fp8-30b/logs'].
- p1: `mose-recover-p1-r3-p1-1x8-pool-1006-1358-5ee-ff2d5`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 5120, 'eta': '7h43m', 'grad': 0.0344, 'ce': 2.728}.
- p0qk: `mose-ug-p0qk-r1-p1-1x8-pool-1006-1347-459e-3-74da9`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 5140, 'eta': '7h39m', 'grad': 0.029, 'ce': 2.658}.
- p2: `mose-recover-p2-r3-p1-2x8-arch-1006-1409-7a8-a7469`; Succeeded; {'active': 0, 'succeeded': 2, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 14300, 'eta': '1m', 'grad': 0.0068, 'ce': 2.508}.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_141212-olmo3-1b-ug-partial-p2-s480-p80-linear-down-rms-fp8-30b', '/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_141212-olmo3-1b-ug-partial-p2-s480-p80-linear-down-rms-fp8-30b/logs'].
- f1: `mose-recover-f1-r2-p1-1x8-pool-1006-1207-08d-af98d`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 7300, 'eta': '5h54m', 'grad': 0.0571, 'ce': 2.658}.
- f2: `mose-recover-f2-r2-p1-1x8-pool-1006-1213-0e1-bdefa`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 7190, 'eta': '6h', 'grad': 0.02, 'ce': 2.655}.
- f3: `mose-recover-f3-r2-p1-1x8-pool-1006-1213-ccc-5d862`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 7290, 'eta': '5h50m', 'grad': 0.0318, 'ce': 2.671}.

2026-10-06 18:21 HKT arch capacity check: [{"nodes": 2, "summary": ["📌 任务调度核心状态：✅ 任务可调度", "  - 总副本数量：2", "  - 可调度数量：2", "  - 不可调度数量：0"]}, {"nodes": 4, "summary": ["📌 任务调度核心状态：✅ 任务可调度", "  - 总副本数量：4", "  - 可调度数量：4", "  - 不可调度数量：0"]}]. Current batch remains ten configs; no new submissions before batch completion.

2026-10-06 18:20 HKT P2QK: verified Succeeded/active0, step14306, final checkpoint saved and Training complete. Exact final W&B synchronization state: started. Completion proof /mnt/shared-storage-user/intern-pretrain-shared02/shijiayang-p/archspace-MoSE/reproduce/olmo-core-backend/scripts/train/UG_Partial_Sharing/completion_p2qk.log. Whole batch still waits for eight pool runs.

2026-10-06 18:20 HKT P2: verified Succeeded/active0, step14306, final checkpoint saved and Training complete. Exact final W&B synchronization state: started. Completion proof /mnt/shared-storage-user/intern-pretrain-shared02/shijiayang-p/archspace-MoSE/reproduce/olmo-core-backend/scripts/train/UG_Partial_Sharing/completion_p2.log. Whole batch still waits for eight pool runs.

18:20 HKT monitoring resumed after local tool/session reset: old monitoring handle41945 disappeared; current scheduler state was queried for the same ten job IDs without replacing any training jobs. A new read-only1800sec loop is running with handle81192, next poll approximately18:50. Last confirmed earlier poll16:00; no uninterrupted monitoring claim for the intervening gap. P2/P2qk Succeeded with final14306 checkpoints; remainingeight pool jobs healthy at5120–8220. Arch16/32 prediction positive18:21, but current batch freeze remains in force.

18:26 HKT: final P2/P2qk W&B uploads both verified Finished syncing. Prepared read-only scheduled history exporter and ten-config batch_segments.json, excluding P2r2 and explicitly separating P1 from strict budget ranking. Python/bash syntax passed; installed local OLMo W&B SDK loaded and unfinished-batch guard rejected the current live batch as expected. Export analysis is not submitted while eight training configs remain unfinished.

## 2026-10-06 18:52 HKT — monitoring

- p0: `mose-ug-p0-r1-p1-1x8-pool-1006-1123-7104-77485485`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 8840, 'eta': '4h33m', 'grad': 0.036, 'ce': 2.616}.
- p4: `mose-recover-p4-r2-p1-1x8-pool-1006-1207-0d3-44098`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 7410, 'eta': '6h13m', 'grad': 0.0797, 'ce': 2.707}.
- p3: `mose-recover-p3-r3-p1-1x8-pool-1006-1228-c38-15484`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 7090, 'eta': '6h25m', 'grad': 0.0219, 'ce': 2.695}.
- p2qk: `mose-ug-p2qk-r1-p1-4x8-arch-1006-1347-8485-7-afb90`; Succeeded; {'active': 0, 'succeeded': 4, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 14300, 'eta': '1m', 'grad': 0.0066, 'ce': 2.504}.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_135053-olmo3-1b-ug-partial-p2qk-s480-p80-linear-qk-silu-down-rms-fp8-30b', '/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_135053-olmo3-1b-ug-partial-p2qk-s480-p80-linear-qk-silu-down-rms-fp8-30b/logs'].
- p1: `mose-recover-p1-r3-p1-1x8-pool-1006-1358-5ee-ff2d5`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 5740, 'eta': '7h11m', 'grad': 0.0415, 'ce': 2.703}.
- p0qk: `mose-ug-p0qk-r1-p1-1x8-pool-1006-1347-459e-3-74da9`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 5780, 'eta': '7h6m', 'grad': 0.0232, 'ce': 2.629}.
- p2: `mose-recover-p2-r3-p1-2x8-arch-1006-1409-7a8-a7469`; Succeeded; {'active': 0, 'succeeded': 2, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 14300, 'eta': '1m', 'grad': 0.0068, 'ce': 2.508}.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_141212-olmo3-1b-ug-partial-p2-s480-p80-linear-down-rms-fp8-30b', '/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_141212-olmo3-1b-ug-partial-p2-s480-p80-linear-down-rms-fp8-30b/logs'].
- f1: `mose-recover-f1-r2-p1-1x8-pool-1006-1207-08d-af98d`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 7930, 'eta': '5h22m', 'grad': 0.0346, 'ce': 2.634}.
- f2: `mose-recover-f2-r2-p1-1x8-pool-1006-1213-0e1-bdefa`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 7820, 'eta': '5h28m', 'grad': 0.0171, 'ce': 2.703}.
- f3: `mose-recover-f3-r2-p1-1x8-pool-1006-1213-ccc-5d862`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 7920, 'eta': '5h18m', 'grad': 0.0356, 'ce': 2.693}.

2026-10-06 18:53 HKT arch capacity check: [{"nodes": 2, "summary": ["📌 任务调度核心状态：✅ 任务可调度", "  - 总副本数量：2", "  - 可调度数量：2", "  - 不可调度数量：0"]}, {"nodes": 4, "summary": ["📌 任务调度核心状态：✅ 任务可调度", "  - 总副本数量：4", "  - 可调度数量：4", "  - 不可调度数量：0"]}]. Current batch remains ten configs; no new submissions before batch completion.

## 2026-10-06 19:22 HKT — monitoring

- p0: `mose-ug-p0-r1-p1-1x8-pool-1006-1123-7104-77485485`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 9470, 'eta': '4h2m', 'grad': 0.0249, 'ce': 2.614}.
- p4: `mose-recover-p4-r2-p1-1x8-pool-1006-1207-0d3-44098`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 7980, 'eta': '5h42m', 'grad': 0.0635, 'ce': 2.66}.
- p3: `mose-recover-p3-r3-p1-1x8-pool-1006-1228-c38-15484`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 7670, 'eta': '5h54m', 'grad': 0.0652, 'ce': 2.684}.
- p2qk: `mose-ug-p2qk-r1-p1-4x8-arch-1006-1347-8485-7-afb90`; Succeeded; {'active': 0, 'succeeded': 4, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 14300, 'eta': '1m', 'grad': 0.0066, 'ce': 2.504}.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_135053-olmo3-1b-ug-partial-p2qk-s480-p80-linear-qk-silu-down-rms-fp8-30b', '/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_135053-olmo3-1b-ug-partial-p2qk-s480-p80-linear-qk-silu-down-rms-fp8-30b/logs'].
- p1: `mose-recover-p1-r3-p1-1x8-pool-1006-1358-5ee-ff2d5`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 6360, 'eta': '6h39m', 'grad': 0.0298, 'ce': 2.656}.
- p0qk: `mose-ug-p0qk-r1-p1-1x8-pool-1006-1347-459e-3-74da9`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 6400, 'eta': '6h35m', 'grad': 0.037, 'ce': 2.653}.
- p2: `mose-recover-p2-r3-p1-2x8-arch-1006-1409-7a8-a7469`; Succeeded; {'active': 0, 'succeeded': 2, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 14300, 'eta': '1m', 'grad': 0.0068, 'ce': 2.508}.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_141212-olmo3-1b-ug-partial-p2-s480-p80-linear-down-rms-fp8-30b', '/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_141212-olmo3-1b-ug-partial-p2-s480-p80-linear-down-rms-fp8-30b/logs'].
- f1: `mose-recover-f1-r2-p1-1x8-pool-1006-1207-08d-af98d`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 8520, 'eta': '4h53m', 'grad': 0.0818, 'ce': 2.583}.
- f2: `mose-recover-f2-r2-p1-1x8-pool-1006-1213-0e1-bdefa`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 8440, 'eta': '4h56m', 'grad': 0.0214, 'ce': 2.587}.
- f3: `mose-recover-f3-r2-p1-1x8-pool-1006-1213-ccc-5d862`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 8520, 'eta': '4h49m', 'grad': 0.0211, 'ce': 2.581}.

19:19 HKT user update: “有卡直接交就行了” overrides batch freeze; validated supplementary combinations may launch immediately while healthy jobs continue. Added F1qk/F2qk/F3qk, changing only Q/K intermediate activations in existing F1/F2/F3 recipes. Strict797,050,880 parameters; native SwiGLU and Down RMSNorm retained. Two targeted tests passed29.36s, including actual meta module counts, full recipe equivalence and8/16/32 globalbatch checks. No core-model changes for these new configs.

19:37 HKT follow-up allocation: F1qk poolr1 never allocated a replica, was stopped and verified active0, and produced no W&B segment. Fresh arch1x8 prediction positive; submitted unique F1qkr2 on that immediately available8-GPU node instead of leaving it idle while the pool attempt queued. All model/data/globalbatch/seed/checkpoint settings unchanged. F3qk continues arch32. F2qk remains one pool8 queued attempt; do not duplicate it. New supplementary configs are outside the original ten-config result manifest.

## 2026-10-06 19:53 HKT — monitoring

- p0: `mose-ug-p0-r1-p1-1x8-pool-1006-1123-7104-77485485`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 10060, 'eta': '3h33m', 'grad': 0.0179, 'ce': 2.607}.
- p4: `mose-recover-p4-r2-p1-1x8-pool-1006-1207-0d3-44098`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 8530, 'eta': '5h13m', 'grad': 0.0424, 'ce': 2.628}.
- p3: `mose-recover-p3-r3-p1-1x8-pool-1006-1228-c38-15484`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 8250, 'eta': '5h23m', 'grad': 0.0537, 'ce': 2.588}.
- p2qk: `mose-ug-p2qk-r1-p1-4x8-arch-1006-1347-8485-7-afb90`; Succeeded; {'active': 0, 'succeeded': 4, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 14300, 'eta': '1m', 'grad': 0.0066, 'ce': 2.504}.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_135053-olmo3-1b-ug-partial-p2qk-s480-p80-linear-qk-silu-down-rms-fp8-30b', '/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_135053-olmo3-1b-ug-partial-p2qk-s480-p80-linear-qk-silu-down-rms-fp8-30b/logs'].
- p1: `mose-recover-p1-r3-p1-1x8-pool-1006-1358-5ee-ff2d5`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 6980, 'eta': '6h7m', 'grad': 0.0201, 'ce': 2.719}.
- p0qk: `mose-ug-p0qk-r1-p1-1x8-pool-1006-1347-459e-3-74da9`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 6990, 'eta': '6h5m', 'grad': 0.0178, 'ce': 2.612}.
- p2: `mose-recover-p2-r3-p1-2x8-arch-1006-1409-7a8-a7469`; Succeeded; {'active': 0, 'succeeded': 2, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 14300, 'eta': '1m', 'grad': 0.0068, 'ce': 2.508}.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_141212-olmo3-1b-ug-partial-p2-s480-p80-linear-down-rms-fp8-30b', '/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_141212-olmo3-1b-ug-partial-p2-s480-p80-linear-down-rms-fp8-30b/logs'].
- f1: `mose-recover-f1-r2-p1-1x8-pool-1006-1207-08d-af98d`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 9120, 'eta': '4h23m', 'grad': 0.1104, 'ce': 2.694}.
- f2: `mose-recover-f2-r2-p1-1x8-pool-1006-1213-0e1-bdefa`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 9020, 'eta': '4h27m', 'grad': 0.0308, 'ce': 2.62}.
- f3: `mose-recover-f3-r2-p1-1x8-pool-1006-1213-ccc-5d862`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 9140, 'eta': '4h18m', 'grad': 0.0314, 'ce': 2.681}.
- f3qk: `mose-ffn-f3qk-r1-p1-4x8-arch-1006-1923-76a7-5350e`; Running; {'active': 4, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 1930, 'eta': '2h41m', 'grad': 0.1086, 'ce': 2.966}.
- f2qk: `mose-ffn-f2qk-r1-p1-1x8-pool-1006-1922-88d3-ed4c5`; Inqueue; {'active': 0, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
- f1qk: `mose-ffn-f1qk-r2-p1-1x8-arch-1006-1937-bfde-3c52a`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 240, 'eta': '11h39m', 'grad': 0.3182, 'ce': 4.775}.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_193938-olmo3-1b-ffn-budget-f1qk-ug384-down768-unshared-qk-silu-ug-linear-down-rms-fp8-30b'].

2026-10-06 20:22 HKT arch capacity check: [{"nodes": 2, "summary": ["📌 任务调度核心状态：❌ 任务不可调度", "  - 总副本数量：2", "  - 可调度数量：0", "  - 不可调度数量：2"]}, {"nodes": 4, "summary": ["📌 任务调度核心状态：❌ 任务不可调度", "  - 总副本数量：4", "  - 可调度数量：0", "  - 不可调度数量：4"]}]. Latest user instruction permits immediate supplementary submissions. Existing healthy jobs stay; pending F2qk has one pool attempt, do not duplicate.

## 2026-10-06 20:23 HKT — monitoring

- p0: `mose-ug-p0-r1-p1-1x8-pool-1006-1123-7104-77485485`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 10680, 'eta': '3h2m', 'grad': 0.0282, 'ce': 2.586}.
- p4: `mose-recover-p4-r2-p1-1x8-pool-1006-1207-0d3-44098`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 9100, 'eta': '4h42m', 'grad': 0.0246, 'ce': 2.596}.
- p3: `mose-recover-p3-r3-p1-1x8-pool-1006-1228-c38-15484`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 8830, 'eta': '4h52m', 'grad': 0.0937, 'ce': 2.614}.
- p2qk: `mose-ug-p2qk-r1-p1-4x8-arch-1006-1347-8485-7-afb90`; Succeeded; {'active': 0, 'succeeded': 4, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 14300, 'eta': '1m', 'grad': 0.0066, 'ce': 2.504}.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_135053-olmo3-1b-ug-partial-p2qk-s480-p80-linear-qk-silu-down-rms-fp8-30b', '/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_135053-olmo3-1b-ug-partial-p2qk-s480-p80-linear-qk-silu-down-rms-fp8-30b/logs'].
- p1: `mose-recover-p1-r3-p1-1x8-pool-1006-1358-5ee-ff2d5`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 7570, 'eta': '5h39m', 'grad': 0.0361, 'ce': 2.583}.
- p0qk: `mose-ug-p0qk-r1-p1-1x8-pool-1006-1347-459e-3-74da9`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 7610, 'eta': '5h35m', 'grad': 0.0271, 'ce': 2.609}.
- p2: `mose-recover-p2-r3-p1-2x8-arch-1006-1409-7a8-a7469`; Succeeded; {'active': 0, 'succeeded': 2, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 14300, 'eta': '1m', 'grad': 0.0068, 'ce': 2.508}.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_141212-olmo3-1b-ug-partial-p2-s480-p80-linear-down-rms-fp8-30b', '/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_141212-olmo3-1b-ug-partial-p2-s480-p80-linear-down-rms-fp8-30b/logs'].
- f1: `mose-recover-f1-r2-p1-1x8-pool-1006-1207-08d-af98d`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 9730, 'eta': '3h52m', 'grad': 0.0234, 'ce': 2.644}.
- f2: `mose-recover-f2-r2-p1-1x8-pool-1006-1213-0e1-bdefa`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 9640, 'eta': '3h56m', 'grad': 0.0162, 'ce': 2.608}.
- f3: `mose-recover-f3-r2-p1-1x8-pool-1006-1213-ccc-5d862`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 9750, 'eta': '3h47m', 'grad': 0.0286, 'ce': 2.636}.
- f3qk: `mose-ffn-f3qk-r1-p1-4x8-arch-1006-1923-76a7-5350e`; Running; {'active': 4, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 4280, 'eta': '2h11m', 'grad': 0.0364, 'ce': 2.737}.
- f2qk: `mose-ffn-f2qk-r1-p1-1x8-pool-1006-1922-88d3-ed4c5`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 280, 'eta': '11h20m', 'grad': 0.2827, 'ce': 4.295}.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_200818-olmo3-1b-ffn-budget-f2qk-ug256-down1024-unshared-qk-silu-ug-linear-down-rms-fp8-30b'].
- f1qk: `mose-ffn-f1qk-r2-p1-1x8-arch-1006-1937-bfde-3c52a`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 830, 'eta': '11h31m', 'grad': 0.2187, 'ce': 3.185}.

## 2026-10-06 20:54 HKT — monitoring

- p0: `mose-ug-p0-r1-p1-1x8-pool-1006-1123-7104-77485485`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 11300, 'eta': '2h31m', 'grad': 0.0167, 'ce': 2.641}.
- p4: `mose-recover-p4-r2-p1-1x8-pool-1006-1207-0d3-44098`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 9660, 'eta': '4h12m', 'grad': 0.0187, 'ce': 2.607}.
- p3: `mose-recover-p3-r3-p1-1x8-pool-1006-1228-c38-15484`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 9410, 'eta': '4h21m', 'grad': 0.0217, 'ce': 2.621}.
- p2qk: `mose-ug-p2qk-r1-p1-4x8-arch-1006-1347-8485-7-afb90`; Succeeded; {'active': 0, 'succeeded': 4, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 14300, 'eta': '1m', 'grad': 0.0066, 'ce': 2.504}.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_135053-olmo3-1b-ug-partial-p2qk-s480-p80-linear-qk-silu-down-rms-fp8-30b', '/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_135053-olmo3-1b-ug-partial-p2qk-s480-p80-linear-qk-silu-down-rms-fp8-30b/logs'].
- p1: `mose-recover-p1-r3-p1-1x8-pool-1006-1358-5ee-ff2d5`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 8190, 'eta': '5h7m', 'grad': 0.0196, 'ce': 2.552}.
- p0qk: `mose-ug-p0qk-r1-p1-1x8-pool-1006-1347-459e-3-74da9`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 8230, 'eta': '5h4m', 'grad': 0.014, 'ce': 2.613}.
- p2: `mose-recover-p2-r3-p1-2x8-arch-1006-1409-7a8-a7469`; Succeeded; {'active': 0, 'succeeded': 2, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 14300, 'eta': '1m', 'grad': 0.0068, 'ce': 2.508}.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_141212-olmo3-1b-ug-partial-p2-s480-p80-linear-down-rms-fp8-30b', '/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_141212-olmo3-1b-ug-partial-p2-s480-p80-linear-down-rms-fp8-30b/logs'].
- f1: `mose-recover-f1-r2-p1-1x8-pool-1006-1207-08d-af98d`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 10340, 'eta': '3h21m', 'grad': 0.0283, 'ce': 2.593}.
- f2: `mose-recover-f2-r2-p1-1x8-pool-1006-1213-0e1-bdefa`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 10250, 'eta': '3h25m', 'grad': 0.0197, 'ce': 2.57}.
- f3: `mose-recover-f3-r2-p1-1x8-pool-1006-1213-ccc-5d862`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 10370, 'eta': '3h16m', 'grad': 0.0202, 'ce': 2.665}.
- f3qk: `mose-ffn-f3qk-r1-p1-4x8-arch-1006-1923-76a7-5350e`; Running; {'active': 4, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 6630, 'eta': '1h40m', 'grad': 0.0152, 'ce': 2.667}.
- f2qk: `mose-ffn-f2qk-r1-p1-1x8-pool-1006-1922-88d3-ed4c5`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 880, 'eta': '11h12m', 'grad': 0.2091, 'ce': 3.268}.
- f1qk: `mose-ffn-f1qk-r2-p1-1x8-arch-1006-1937-bfde-3c52a`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 1420, 'eta': '11h1m', 'grad': 0.1971, 'ce': 3.106}.

2026-10-06 20:55 HKT arch capacity check: [{"nodes": 2, "summary": ["📌 任务调度核心状态：❌ 任务不可调度", "  - 总副本数量：2", "  - 可调度数量：1", "  - 不可调度数量：1"]}, {"nodes": 4, "summary": ["📌 任务调度核心状态：❌ 任务不可调度", "  - 总副本数量：4", "  - 可调度数量：1", "  - 不可调度数量：3"]}]. Immediate supplementary submissions permitted; healthy jobs stay on their current allocation.

Submitted P3QK: `mose-ug-p3qk-r1-p1-1x8-pool-1006-2111-2dcc-7-ec6b7`, pool 1x8, priority1; script `/mnt/shared-storage-gpfs2/intern-pretrain-shared02/shijiayang-p/archspace-MoSE/reproduce/olmo-core-backend/scripts/train/UG_Partial_Sharing/run_p3qk.sh`.

Submitted P4QK: `mose-ug-p4qk-r1-p1-1x8-pool-1006-2111-0416-8-5686b`, pool 1x8, priority1; script `/mnt/shared-storage-gpfs2/intern-pretrain-shared02/shijiayang-p/archspace-MoSE/reproduce/olmo-core-backend/scripts/train/UG_Partial_Sharing/run_p4qk.sh`.

## 2026-10-06 21:24 HKT — monitoring

- p0: `mose-ug-p0-r1-p1-1x8-pool-1006-1123-7104-77485485`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 11910, 'eta': '2h', 'grad': 0.0296, 'ce': 2.626}.
- p4: `mose-recover-p4-r2-p1-1x8-pool-1006-1207-0d3-44098`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 10220, 'eta': '3h41m', 'grad': 0.0734, 'ce': 2.618}.
- p3: `mose-recover-p3-r3-p1-1x8-pool-1006-1228-c38-15484`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 9980, 'eta': '3h51m', 'grad': 0.0756, 'ce': 2.708}.
- p2qk: `mose-ug-p2qk-r1-p1-4x8-arch-1006-1347-8485-7-afb90`; Succeeded; {'active': 0, 'succeeded': 4, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 14300, 'eta': '1m', 'grad': 0.0066, 'ce': 2.504}.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_135053-olmo3-1b-ug-partial-p2qk-s480-p80-linear-qk-silu-down-rms-fp8-30b', '/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_135053-olmo3-1b-ug-partial-p2qk-s480-p80-linear-qk-silu-down-rms-fp8-30b/logs'].
- p1: `mose-recover-p1-r3-p1-1x8-pool-1006-1358-5ee-ff2d5`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 8790, 'eta': '4h37m', 'grad': 0.0161, 'ce': 2.606}.
- p0qk: `mose-ug-p0qk-r1-p1-1x8-pool-1006-1347-459e-3-74da9`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 8840, 'eta': '4h33m', 'grad': 0.0247, 'ce': 2.612}.
- p2: `mose-recover-p2-r3-p1-2x8-arch-1006-1409-7a8-a7469`; Succeeded; {'active': 0, 'succeeded': 2, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 14300, 'eta': '1m', 'grad': 0.0068, 'ce': 2.508}.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_141212-olmo3-1b-ug-partial-p2-s480-p80-linear-down-rms-fp8-30b', '/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_141212-olmo3-1b-ug-partial-p2-s480-p80-linear-down-rms-fp8-30b/logs'].
- p3qk: `mose-ug-p3qk-r1-p1-1x8-pool-1006-2111-2dcc-7-ec6b7`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 160, 'eta': '12h7m', 'grad': 0.4811, 'ce': 5.737}.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_211408-olmo3-1b-ug-partial-p3qk-s320-p224-linear-qk-silu-down-rms-fp8-30b'].
- p4qk: `mose-ug-p4qk-r1-p1-1x8-pool-1006-2111-0416-8-5686b`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 150, 'eta': '12h8m', 'grad': 0.6132, 'ce': 5.814}.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_211424-olmo3-1b-ug-partial-p4qk-s160-p368-linear-qk-silu-down-rms-fp8-30b'].
- f1: `mose-recover-f1-r2-p1-1x8-pool-1006-1207-08d-af98d`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 10940, 'eta': '2h50m', 'grad': 0.0296, 'ce': 2.61}.
- f2: `mose-recover-f2-r2-p1-1x8-pool-1006-1213-0e1-bdefa`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 10850, 'eta': '2h55m', 'grad': 0.0225, 'ce': 2.633}.
- f3: `mose-recover-f3-r2-p1-1x8-pool-1006-1213-ccc-5d862`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 10990, 'eta': '2h45m', 'grad': 0.0218, 'ce': 2.619}.
- f3qk: `mose-ffn-f3qk-r1-p1-4x8-arch-1006-1923-76a7-5350e`; Running; {'active': 4, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 8970, 'eta': '1h10m', 'grad': 0.0184, 'ce': 2.659}.
- f2qk: `mose-ffn-f2qk-r1-p1-1x8-pool-1006-1922-88d3-ed4c5`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 1480, 'eta': '10h43m', 'grad': 0.1797, 'ce': 3.068}.
- f1qk: `mose-ffn-f1qk-r2-p1-1x8-arch-1006-1937-bfde-3c52a`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 1990, 'eta': '10h32m', 'grad': 0.1085, 'ce': 2.943}.
- c0s2: `mose-r512-c0s2-r1-p1-1x8-pool-1006-2106-ea9c-f2516`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 260, 'eta': '11h26m', 'grad': 0.2866, 'ce': 4.57}.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_210958-olmo3-1b-r512matched-c0-seed12537-data34521-fp8-30b'].
- c6s2: `mose-r512-c6s2-r1-p1-1x8-pool-1006-2107-3d78-52e71`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 250, 'eta': '11h49m', 'grad': 0.3802, 'ce': 4.744}.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_211039-olmo3-1b-r512matched-c6-seed12537-data34521-fp8-30b'].
- c0s3: `mose-r512-c0s3-r1-p1-1x8-pool-1006-2107-2f04-80461`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 240, 'eta': '11h31m', 'grad': 0.4029, 'ce': 4.755}.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_211054-olmo3-1b-r512matched-c0-seed12538-data34521-fp8-30b'].
- c6s3: `mose-r512-c6s3-r1-p1-1x8-pool-1006-2108-31a7-fc86e`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 240, 'eta': '11h38m', 'grad': 0.3568, 'ce': 4.8}.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_211118-olmo3-1b-r512matched-c6-seed12538-data34521-fp8-30b'].

## 21:24 HKT — startup audit

- {"variant": "p3qk", "job_id": "mose-ug-p3qk-r1-p1-1x8-pool-1006-2111-2dcc-7-ec6b7", "exit_code": 0, "mesh": ["(dp_replicate=1, dp_shard=8)"], "segments": ["/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_211408-olmo3-1b-ug-partial-p3qk-s320-p224-linear-qk-silu-down-rms-fp8-30b"], "step0_saved": false, "nonfinite_ce": false, "oom": false, "traceback": false, "proof": "/mnt/shared-storage-user/intern-pretrain-shared02/shijiayang-p/archspace-MoSE/reproduce/olmo-core-backend/scripts/train/UG_Partial_Sharing/startup_p3qk.log", "observed_at": "2026-10-06T21:25:42.046154+08:00"}
- {"variant": "p4qk", "job_id": "mose-ug-p4qk-r1-p1-1x8-pool-1006-2111-0416-8-5686b", "exit_code": 0, "mesh": ["(dp_replicate=1, dp_shard=8)"], "segments": ["/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_211424-olmo3-1b-ug-partial-p4qk-s160-p368-linear-qk-silu-down-rms-fp8-30b"], "step0_saved": false, "nonfinite_ce": false, "oom": false, "traceback": false, "proof": "/mnt/shared-storage-user/intern-pretrain-shared02/shijiayang-p/archspace-MoSE/reproduce/olmo-core-backend/scripts/train/UG_Partial_Sharing/startup_p4qk.log", "observed_at": "2026-10-06T21:25:41.648401+08:00"}

21:27 startup audit correction: async checkpoint success is logged as "Checkpoint for step 0 saved successfully". All new variants in this registry have that explicit success line, fresh initialization, (dp_replicate=1,dp_shard=8), finite CE, and no OOM/Traceback. Earlier false flags were parser mismatches; actual checkpoint writes succeeded.

## 21:27 HKT — consolidated update

19 registered configurations:2 fully completed/synced (P2/P2qk),17 Running with finite logged CE and zero failed/stopped replicas. Six new pool1x8 priority1 jobs are all training: P3qk/P4qk and C0/C6 paired init-seed12537/12538 replications. arch16/32 still not immediately schedulable. F3qk was at8970/14306, ETA1h10m at21:24. Next unified monitor poll approximately21:54; local session76585 remains active. No GPU duplicate, migration, high-priority submission, or destructive cleanup.

## 2026-10-06 21:55 HKT — monitoring

- p0: `mose-ug-p0-r1-p1-1x8-pool-1006-1123-7104-77485485`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 12490, 'eta': '1h31m', 'grad': 0.0151, 'ce': 2.53}.
- p4: `mose-recover-p4-r2-p1-1x8-pool-1006-1207-0d3-44098`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 10800, 'eta': '3h10m', 'grad': 0.0357, 'ce': 2.609}.
- p3: `mose-recover-p3-r3-p1-1x8-pool-1006-1228-c38-15484`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 10530, 'eta': '3h22m', 'grad': 0.0686, 'ce': 2.612}.
- p2qk: `mose-ug-p2qk-r1-p1-4x8-arch-1006-1347-8485-7-afb90`; Succeeded; {'active': 0, 'succeeded': 4, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 14300, 'eta': '1m', 'grad': 0.0066, 'ce': 2.504}.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_135053-olmo3-1b-ug-partial-p2qk-s480-p80-linear-qk-silu-down-rms-fp8-30b', '/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_135053-olmo3-1b-ug-partial-p2qk-s480-p80-linear-qk-silu-down-rms-fp8-30b/logs'].
- p1: `mose-recover-p1-r3-p1-1x8-pool-1006-1358-5ee-ff2d5`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 9410, 'eta': '4h6m', 'grad': 0.0263, 'ce': 2.619}.
- p0qk: `mose-ug-p0qk-r1-p1-1x8-pool-1006-1347-459e-3-74da9`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 9460, 'eta': '4h2m', 'grad': 0.0221, 'ce': 2.585}.
- p2: `mose-recover-p2-r3-p1-2x8-arch-1006-1409-7a8-a7469`; Succeeded; {'active': 0, 'succeeded': 2, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 14300, 'eta': '1m', 'grad': 0.0068, 'ce': 2.508}.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_141212-olmo3-1b-ug-partial-p2-s480-p80-linear-down-rms-fp8-30b', '/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_141212-olmo3-1b-ug-partial-p2-s480-p80-linear-down-rms-fp8-30b/logs'].
- p3qk: `mose-ug-p3qk-r1-p1-1x8-pool-1006-2111-2dcc-7-ec6b7`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 730, 'eta': '12h7m', 'grad': 0.1838, 'ce': 3.266}.
- p4qk: `mose-ug-p4qk-r1-p1-1x8-pool-1006-2111-0416-8-5686b`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 720, 'eta': '12h9m', 'grad': 0.2246, 'ce': 3.402}.
- f1: `mose-recover-f1-r2-p1-1x8-pool-1006-1207-08d-af98d`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 11530, 'eta': '2h21m', 'grad': 0.1425, 'ce': 2.621}.
- f2: `mose-recover-f2-r2-p1-1x8-pool-1006-1213-0e1-bdefa`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 11470, 'eta': '2h23m', 'grad': 0.0155, 'ce': 2.565}.
- f3: `mose-recover-f3-r2-p1-1x8-pool-1006-1213-ccc-5d862`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 11580, 'eta': '2h16m', 'grad': 0.0182, 'ce': 2.559}.
- f3qk: `mose-ffn-f3qk-r1-p1-4x8-arch-1006-1923-76a7-5350e`; Running; {'active': 4, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 11330, 'eta': '39m', 'grad': 0.023, 'ce': 2.678}.
- f2qk: `mose-ffn-f2qk-r1-p1-1x8-pool-1006-1922-88d3-ed4c5`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 2070, 'eta': '10h20m', 'grad': 0.0818, 'ce': 2.978}.
- f1qk: `mose-ffn-f1qk-r2-p1-1x8-arch-1006-1937-bfde-3c52a`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 2590, 'eta': '10h6m', 'grad': 0.0717, 'ce': 2.885}.
- c0s2: `mose-r512-c0s2-r1-p1-1x8-pool-1006-2106-ea9c-f2516`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 860, 'eta': '11h20m', 'grad': 0.1577, 'ce': 3.218}.
- c6s2: `mose-r512-c6s2-r1-p1-1x8-pool-1006-2107-3d78-52e71`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 830, 'eta': '11h42m', 'grad': 0.1621, 'ce': 3.181}.
- c0s3: `mose-r512-c0s3-r1-p1-1x8-pool-1006-2107-2f04-80461`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 840, 'eta': '11h24m', 'grad': 0.2349, 'ce': 3.167}.
- c6s3: `mose-r512-c6s3-r1-p1-1x8-pool-1006-2108-31a7-fc86e`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 830, 'eta': '11h31m', 'grad': 0.1408, 'ce': 3.179}.

## 21:59 HKT — complete histories recovered

19 training configs:2 completed/synced,17 Running without failed/stopped replicas at21:55. F3qk11330/14306, ETA39min. Read-only P2/P2qk report job Succeeded/active0; full JSON SHA256 matched, both14306-step histories complete with empty anomaly mask. Primary final1000 means: P2=2.484496541261673, P2qk=2.4812729761600494 (delta−0.003223565101623471). Saved full statistics/provenance in P2_P2QK_FULL_TRAIN_CE.json and P2_P2QK_RESULTS.md. Current arch16/32 and pool8 predictions were not schedulable at21:57; no additional training submission. Next unified check about22:25, session76585 remains live.

## 2026-10-06 22:26 HKT — monitoring

- p0: `mose-ug-p0-r1-p1-1x8-pool-1006-1123-7104-77485485`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 13100, 'eta': '1h1m', 'grad': 0.0169, 'ce': 2.539}.
- p4: `mose-recover-p4-r2-p1-1x8-pool-1006-1207-0d3-44098`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 11370, 'eta': '2h39m', 'grad': 0.0524, 'ce': 2.617}.
- p3: `mose-recover-p3-r3-p1-1x8-pool-1006-1228-c38-15484`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 11110, 'eta': '2h51m', 'grad': 0.0875, 'ce': 2.626}.
- p2qk: `mose-ug-p2qk-r1-p1-4x8-arch-1006-1347-8485-7-afb90`; Succeeded; {'active': 0, 'succeeded': 4, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 14300, 'eta': '1m', 'grad': 0.0066, 'ce': 2.504}.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_135053-olmo3-1b-ug-partial-p2qk-s480-p80-linear-qk-silu-down-rms-fp8-30b', '/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_135053-olmo3-1b-ug-partial-p2qk-s480-p80-linear-qk-silu-down-rms-fp8-30b/logs'].
- p1: `mose-recover-p1-r3-p1-1x8-pool-1006-1358-5ee-ff2d5`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 10010, 'eta': '3h36m', 'grad': 0.0268, 'ce': 2.632}.
- p0qk: `mose-ug-p0qk-r1-p1-1x8-pool-1006-1347-459e-3-74da9`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 10050, 'eta': '3h33m', 'grad': 0.0281, 'ce': 2.601}.
- p2: `mose-recover-p2-r3-p1-2x8-arch-1006-1409-7a8-a7469`; Succeeded; {'active': 0, 'succeeded': 2, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 14300, 'eta': '1m', 'grad': 0.0068, 'ce': 2.508}.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_141212-olmo3-1b-ug-partial-p2-s480-p80-linear-down-rms-fp8-30b', '/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_141212-olmo3-1b-ug-partial-p2-s480-p80-linear-down-rms-fp8-30b/logs'].
- p3qk: `mose-ug-p3qk-r1-p1-1x8-pool-1006-2111-2dcc-7-ec6b7`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 1300, 'eta': '11h36m', 'grad': 0.1206, 'ce': 3.147}.
- p4qk: `mose-ug-p4qk-r1-p1-1x8-pool-1006-2111-0416-8-5686b`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 1290, 'eta': '11h38m', 'grad': 0.1087, 'ce': 3.064}.
- f1: `mose-recover-f1-r2-p1-1x8-pool-1006-1207-08d-af98d`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 12150, 'eta': '1h49m', 'grad': 0.0387, 'ce': 2.625}.
- f2: `mose-recover-f2-r2-p1-1x8-pool-1006-1213-0e1-bdefa`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 12060, 'eta': '1h54m', 'grad': 0.0227, 'ce': 2.609}.
- f3: `mose-recover-f3-r2-p1-1x8-pool-1006-1213-ccc-5d862`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 12210, 'eta': '1h45m', 'grad': 0.0167, 'ce': 2.582}.
- f3qk: `mose-ffn-f3qk-r1-p1-4x8-arch-1006-1923-76a7-5350e`; Running; {'active': 4, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 13630, 'eta': '9m', 'grad': 0.0113, 'ce': 2.471}.
- f2qk: `mose-ffn-f2qk-r1-p1-1x8-pool-1006-1922-88d3-ed4c5`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 2690, 'eta': '9h47m', 'grad': 0.0414, 'ce': 2.923}.
- f1qk: `mose-ffn-f1qk-r2-p1-1x8-arch-1006-1937-bfde-3c52a`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 3190, 'eta': '9h33m', 'grad': 0.0402, 'ce': 2.82}.
- c0s2: `mose-r512-c0s2-r1-p1-1x8-pool-1006-2106-ea9c-f2516`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 1460, 'eta': '10h50m', 'grad': 0.1522, 'ce': 3.076}.
- c6s2: `mose-r512-c6s2-r1-p1-1x8-pool-1006-2107-3d78-52e71`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 1410, 'eta': '11h14m', 'grad': 0.1773, 'ce': 3.054}.
- c0s3: `mose-r512-c0s3-r1-p1-1x8-pool-1006-2107-2f04-80461`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 1440, 'eta': '10h54m', 'grad': 0.1337, 'ce': 3.043}.
- c6s3: `mose-r512-c6s3-r1-p1-1x8-pool-1006-2108-31a7-fc86e`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 1420, 'eta': '11h2m', 'grad': 0.1682, 'ce': 3.095}.

## 22:30 HKT — monitor/capacity update

22:26 unified poll:19 training configs,2 completed/synced and17 Running with finite latest CE and zero failed/stopped replicas. F3qk13630/14306 ETA9min; P0 ETA1h1m; F1/F2/F3 each approximately1h45m–1h54m. The six new pool runs have reached roughly1290–1460 steps and remain healthy. Historical C0/C6 scheduler queries returned no authoritative status/log body, so no reference completion was inferred and original-seed aggregation remains deferred. Existing complete historical results remain intact. Next scheduled monitor poll approximately22:56.

## 2026-10-06 22:56 HKT — monitoring

- p0: `mose-ug-p0-r1-p1-1x8-pool-1006-1123-7104-77485485`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 13720, 'eta': '30m', 'grad': 0.0115, 'ce': 2.444}.
- p4: `mose-recover-p4-r2-p1-1x8-pool-1006-1207-0d3-44098`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 11940, 'eta': '2h8m', 'grad': 0.0255, 'ce': 2.614}.
- p3: `mose-recover-p3-r3-p1-1x8-pool-1006-1228-c38-15484`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 11690, 'eta': '2h20m', 'grad': 0.0614, 'ce': 2.612}.
- p2qk: `mose-ug-p2qk-r1-p1-4x8-arch-1006-1347-8485-7-afb90`; Succeeded; {'active': 0, 'succeeded': 4, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 14300, 'eta': '1m', 'grad': 0.0066, 'ce': 2.504}.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_135053-olmo3-1b-ug-partial-p2qk-s480-p80-linear-qk-silu-down-rms-fp8-30b', '/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_135053-olmo3-1b-ug-partial-p2qk-s480-p80-linear-qk-silu-down-rms-fp8-30b/logs'].
- p1: `mose-recover-p1-r3-p1-1x8-pool-1006-1358-5ee-ff2d5`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 10620, 'eta': '3h5m', 'grad': 0.0222, 'ce': 2.623}.
- p0qk: `mose-ug-p0qk-r1-p1-1x8-pool-1006-1347-459e-3-74da9`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 10670, 'eta': '3h2m', 'grad': 0.0148, 'ce': 2.517}.
- p2: `mose-recover-p2-r3-p1-2x8-arch-1006-1409-7a8-a7469`; Succeeded; {'active': 0, 'succeeded': 2, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 14300, 'eta': '1m', 'grad': 0.0068, 'ce': 2.508}.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_141212-olmo3-1b-ug-partial-p2-s480-p80-linear-down-rms-fp8-30b', '/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_141212-olmo3-1b-ug-partial-p2-s480-p80-linear-down-rms-fp8-30b/logs'].
- p3qk: `mose-ug-p3qk-r1-p1-1x8-pool-1006-2111-2dcc-7-ec6b7`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 1880, 'eta': '11h5m', 'grad': 0.1181, 'ce': 2.989}.
- p4qk: `mose-ug-p4qk-r1-p1-1x8-pool-1006-2111-0416-8-5686b`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 1870, 'eta': '11h6m', 'grad': 0.0839, 'ce': 2.963}.
- f1: `mose-recover-f1-r2-p1-1x8-pool-1006-1207-08d-af98d`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 12740, 'eta': '1h20m', 'grad': 0.0206, 'ce': 2.449}.
- f2: `mose-recover-f2-r2-p1-1x8-pool-1006-1213-0e1-bdefa`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 12650, 'eta': '1h24m', 'grad': 0.0169, 'ce': 2.591}.
- f3: `mose-recover-f3-r2-p1-1x8-pool-1006-1213-ccc-5d862`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 12810, 'eta': '1h15m', 'grad': 0.0136, 'ce': 2.522}.
- f3qk: `mose-ffn-f3qk-r1-p1-4x8-arch-1006-1923-76a7-5350e`; Succeeded; {'active': 0, 'succeeded': 4, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 14300, 'eta': '1m', 'grad': 0.0071, 'ce': 2.5}.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_192653-olmo3-1b-ffn-budget-f3qk-ug480-down672-shared-qk-silu-ug-linear-down-rms-fp8-30b', '/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_192653-olmo3-1b-ffn-budget-f3qk-ug480-down672-shared-qk-silu-ug-linear-down-rms-fp8-30b/logs'].
- f2qk: `mose-ffn-f2qk-r1-p1-1x8-pool-1006-1922-88d3-ed4c5`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 3300, 'eta': '9h15m', 'grad': 0.0819, 'ce': 2.856}.
- f1qk: `mose-ffn-f1qk-r2-p1-1x8-arch-1006-1937-bfde-3c52a`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 3790, 'eta': '9h2m', 'grad': 0.0639, 'ce': 2.724}.
- c0s2: `mose-r512-c0s2-r1-p1-1x8-pool-1006-2106-ea9c-f2516`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 2050, 'eta': '10h28m', 'grad': 0.1272, 'ce': 2.963}.
- c6s2: `mose-r512-c6s2-r1-p1-1x8-pool-1006-2107-3d78-52e71`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 1990, 'eta': '10h44m', 'grad': 0.0492, 'ce': 2.93}.
- c0s3: `mose-r512-c0s3-r1-p1-1x8-pool-1006-2107-2f04-80461`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 2020, 'eta': '10h32m', 'grad': 0.0858, 'ce': 2.971}.
- c6s3: `mose-r512-c6s3-r1-p1-1x8-pool-1006-2108-31a7-fc86e`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 1990, 'eta': '10h33m', 'grad': 0.0954, 'ce': 2.93}.

## 2026-10-06 23:33 HKT — monitoring

- p0: `mose-ug-p0-r1-p1-1x8-pool-1006-1123-7104-77485485`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 14300, 'eta': '1m', 'grad': 0.0075, 'ce': 2.503}.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_112904-olmo3-1b-ug-partial-p0-s0-p512-linear-down-rms-fp8-30b', '/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_112904-olmo3-1b-ug-partial-p0-s0-p512-linear-down-rms-fp8-30b/logs'].
- p4: `mose-recover-p4-r2-p1-1x8-pool-1006-1207-0d3-44098`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 12490, 'eta': '1h38m', 'grad': 0.0306, 'ce': 2.537}.
- p3: `mose-recover-p3-r3-p1-1x8-pool-1006-1228-c38-15484`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 12300, 'eta': '1h47m', 'grad': 0.0137, 'ce': 2.527}.
- p2qk: `mose-ug-p2qk-r1-p1-4x8-arch-1006-1347-8485-7-afb90`; Succeeded; {'active': 0, 'succeeded': 4, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 14300, 'eta': '1m', 'grad': 0.0066, 'ce': 2.504}.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_135053-olmo3-1b-ug-partial-p2qk-s480-p80-linear-qk-silu-down-rms-fp8-30b', '/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_135053-olmo3-1b-ug-partial-p2qk-s480-p80-linear-qk-silu-down-rms-fp8-30b/logs'].
- p1: `mose-recover-p1-r3-p1-1x8-pool-1006-1358-5ee-ff2d5`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 11270, 'eta': '2h33m', 'grad': 0.0204, 'ce': 2.536}.
- p0qk: `mose-ug-p0qk-r1-p1-1x8-pool-1006-1347-459e-3-74da9`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 11350, 'eta': '2h28m', 'grad': 0.0201, 'ce': 2.645}.
- p2: `mose-recover-p2-r3-p1-2x8-arch-1006-1409-7a8-a7469`; Succeeded; {'active': 0, 'succeeded': 2, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 14300, 'eta': '1m', 'grad': 0.0068, 'ce': 2.508}.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_141212-olmo3-1b-ug-partial-p2-s480-p80-linear-down-rms-fp8-30b', '/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_141212-olmo3-1b-ug-partial-p2-s480-p80-linear-down-rms-fp8-30b/logs'].
- p3qk: `mose-ug-p3qk-r1-p1-1x8-pool-1006-2111-2dcc-7-ec6b7`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 2490, 'eta': '10h31m', 'grad': 0.0382, 'ce': 2.834}.
- p4qk: `mose-ug-p4qk-r1-p1-1x8-pool-1006-2111-0416-8-5686b`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 2490, 'eta': '10h32m', 'grad': 0.0322, 'ce': 2.833}.
- f1: `mose-recover-f1-r2-p1-1x8-pool-1006-1207-08d-af98d`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 13410, 'eta': '46m', 'grad': 0.0157, 'ce': 2.496}.
- f2: `mose-recover-f2-r2-p1-1x8-pool-1006-1213-0e1-bdefa`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 13350, 'eta': '49m', 'grad': 0.0137, 'ce': 2.55}.
- f3: `mose-recover-f3-r2-p1-1x8-pool-1006-1213-ccc-5d862`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 13490, 'eta': '41m', 'grad': 0.0141, 'ce': 2.529}.
- f3qk: `mose-ffn-f3qk-r1-p1-4x8-arch-1006-1923-76a7-5350e`; Succeeded; {'active': 0, 'succeeded': 4, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 14300, 'eta': '1m', 'grad': 0.0071, 'ce': 2.5}.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_192653-olmo3-1b-ffn-budget-f3qk-ug480-down672-shared-qk-silu-ug-linear-down-rms-fp8-30b', '/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_192653-olmo3-1b-ffn-budget-f3qk-ug480-down672-shared-qk-silu-ug-linear-down-rms-fp8-30b/logs'].
- f2qk: `mose-ffn-f2qk-r1-p1-1x8-pool-1006-1922-88d3-ed4c5`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 3990, 'eta': '8h39m', 'grad': 0.0277, 'ce': 2.769}.
- f1qk: `mose-ffn-f1qk-r2-p1-1x8-arch-1006-1937-bfde-3c52a`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 4470, 'eta': '8h26m', 'grad': 0.0594, 'ce': 2.779}.
- f4qk: `mose-ffn-f4qk-r1-p1-2x8-arch-1006-2300-6d41-46be5`; Running; {'active': 2, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 1000, 'eta': '5h47m', 'grad': 0.1538, 'ce': 3.148}.
- f5qk: `mose-ffn-f5qk-r1-p1-2x8-arch-1006-2300-f278-d8613`; Running; {'active': 2, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 990, 'eta': '5h32m', 'grad': 0.2268, 'ce': 3.224}.
- c0s2: `mose-r512-c0s2-r1-p1-1x8-pool-1006-2106-ea9c-f2516`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 2760, 'eta': '9h48m', 'grad': 0.0251, 'ce': 2.819}.
- c6s2: `mose-r512-c6s2-r1-p1-1x8-pool-1006-2107-3d78-52e71`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 2670, 'eta': '10h12m', 'grad': 0.0214, 'ce': 2.776}.
- c0s3: `mose-r512-c0s3-r1-p1-1x8-pool-1006-2107-2f04-80461`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 2740, 'eta': '9h52m', 'grad': 0.0828, 'ce': 2.871}.
- c6s3: `mose-r512-c6s3-r1-p1-1x8-pool-1006-2108-31a7-fc86e`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 2730, 'eta': '9h58m', 'grad': 0.0684, 'ce': 2.807}.

P0 completion verified2026-10-06 23:33 HKT: Succeeded/active0/succeeded1, full14306, final checkpoint saved23:24:59, Training complete23:25:02. Exact W&B upload launched; full Train CE analysis pending capacity.

## 2026-10-06 23:56 HKT — monitoring

- p0: `mose-ug-p0-r1-p1-1x8-pool-1006-1123-7104-77485485`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 14300, 'eta': '1m', 'grad': 0.0075, 'ce': 2.503}.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_112904-olmo3-1b-ug-partial-p0-s0-p512-linear-down-rms-fp8-30b', '/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_112904-olmo3-1b-ug-partial-p0-s0-p512-linear-down-rms-fp8-30b/logs'].
- p4: `mose-recover-p4-r2-p1-1x8-pool-1006-1207-0d3-44098`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 13010, 'eta': '1h11m', 'grad': 0.029, 'ce': 2.479}.
- p3: `mose-recover-p3-r3-p1-1x8-pool-1006-1228-c38-15484`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 12810, 'eta': '1h20m', 'grad': 0.115, 'ce': 2.525}.
- p2qk: `mose-ug-p2qk-r1-p1-4x8-arch-1006-1347-8485-7-afb90`; Succeeded; {'active': 0, 'succeeded': 4, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 14300, 'eta': '1m', 'grad': 0.0066, 'ce': 2.504}.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_135053-olmo3-1b-ug-partial-p2qk-s480-p80-linear-qk-silu-down-rms-fp8-30b', '/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_135053-olmo3-1b-ug-partial-p2qk-s480-p80-linear-qk-silu-down-rms-fp8-30b/logs'].
- p1: `mose-recover-p1-r3-p1-1x8-pool-1006-1358-5ee-ff2d5`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 11830, 'eta': '2h5m', 'grad': 0.0244, 'ce': 2.611}.
- p0qk: `mose-ug-p0qk-r1-p1-1x8-pool-1006-1347-459e-3-74da9`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 11890, 'eta': '2h1m', 'grad': 0.0312, 'ce': 2.571}.
- p2: `mose-recover-p2-r3-p1-2x8-arch-1006-1409-7a8-a7469`; Succeeded; {'active': 0, 'succeeded': 2, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 14300, 'eta': '1m', 'grad': 0.0068, 'ce': 2.508}.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_141212-olmo3-1b-ug-partial-p2-s480-p80-linear-down-rms-fp8-30b', '/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_141212-olmo3-1b-ug-partial-p2-s480-p80-linear-down-rms-fp8-30b/logs'].
- p3qk: `mose-ug-p3qk-r1-p1-1x8-pool-1006-2111-2dcc-7-ec6b7`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 2990, 'eta': '10h4m', 'grad': 0.0435, 'ce': 2.829}.
- p4qk: `mose-ug-p4qk-r1-p1-1x8-pool-1006-2111-0416-8-5686b`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 2990, 'eta': '10h5m', 'grad': 0.0409, 'ce': 2.831}.
- f1: `mose-recover-f1-r2-p1-1x8-pool-1006-1207-08d-af98d`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 13930, 'eta': '19m', 'grad': 0.014, 'ce': 2.501}.
- f2: `mose-recover-f2-r2-p1-1x8-pool-1006-1213-0e1-bdefa`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 13850, 'eta': '23m', 'grad': 0.0138, 'ce': 2.455}.
- f3: `mose-recover-f3-r2-p1-1x8-pool-1006-1213-ccc-5d862`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 13990, 'eta': '16m', 'grad': 0.0178, 'ce': 2.53}.
- f3qk: `mose-ffn-f3qk-r1-p1-4x8-arch-1006-1923-76a7-5350e`; Succeeded; {'active': 0, 'succeeded': 4, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 14300, 'eta': '1m', 'grad': 0.0071, 'ce': 2.5}.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_192653-olmo3-1b-ffn-budget-f3qk-ug480-down672-shared-qk-silu-ug-linear-down-rms-fp8-30b', '/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_192653-olmo3-1b-ffn-budget-f3qk-ug480-down672-shared-qk-silu-ug-linear-down-rms-fp8-30b/logs'].
- f2qk: `mose-ffn-f2qk-r1-p1-1x8-pool-1006-1922-88d3-ed4c5`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 4490, 'eta': '8h14m', 'grad': 0.0238, 'ce': 2.779}.
- f1qk: `mose-ffn-f1qk-r2-p1-1x8-arch-1006-1937-bfde-3c52a`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 4960, 'eta': '8h1m', 'grad': 0.1119, 'ce': 2.695}.
- f4qk: `mose-ffn-f4qk-r1-p1-2x8-arch-1006-2300-6d41-46be5`; Running; {'active': 2, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 1940, 'eta': '5h18m', 'grad': 0.0889, 'ce': 2.975}.
- f5qk: `mose-ffn-f5qk-r1-p1-2x8-arch-1006-2300-f278-d8613`; Running; {'active': 2, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 1960, 'eta': '5h11m', 'grad': 0.0646, 'ce': 3.022}.
- c0s2: `mose-r512-c0s2-r1-p1-1x8-pool-1006-2106-ea9c-f2516`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 3230, 'eta': '9h25m', 'grad': 0.0589, 'ce': 2.774}.
- c6s2: `mose-r512-c6s2-r1-p1-1x8-pool-1006-2107-3d78-52e71`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 3130, 'eta': '9h48m', 'grad': 0.0192, 'ce': 2.855}.
- c0s3: `mose-r512-c0s3-r1-p1-1x8-pool-1006-2107-2f04-80461`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 3200, 'eta': '9h28m', 'grad': 0.0934, 'ce': 2.697}.
- c6s3: `mose-r512-c6s3-r1-p1-1x8-pool-1006-2108-31a7-fc86e`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 3170, 'eta': '9h36m', 'grad': 0.021, 'ce': 2.792}.
- c2s2: `mose-r512-c2s2-r1-p1-1x8-pool-1006-2338-833a-1f47a`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 270, 'eta': '11h32m', 'grad': 0.2858, 'ce': 4.494}.
- c2s3: `mose-r512-c2s3-r1-p1-1x8-pool-1006-2338-d226-b5503`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 240, 'eta': '11h46m', 'grad': 0.2834, 'ce': 4.783}.

## 2026-10-07 00:26 HKT — monitoring

- p0: `mose-ug-p0-r1-p1-1x8-pool-1006-1123-7104-77485485`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 14300, 'eta': '1m', 'grad': 0.0075, 'ce': 2.503}.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_112904-olmo3-1b-ug-partial-p0-s0-p512-linear-down-rms-fp8-30b', '/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_112904-olmo3-1b-ug-partial-p0-s0-p512-linear-down-rms-fp8-30b/logs'].
- p4: `mose-recover-p4-r2-p1-1x8-pool-1006-1207-0d3-44098`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 13560, 'eta': '41m', 'grad': 0.0201, 'ce': 2.533}.
- p3: `mose-recover-p3-r3-p1-1x8-pool-1006-1228-c38-15484`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 13370, 'eta': '50m', 'grad': 0.0151, 'ce': 2.549}.
- p2qk: `mose-ug-p2qk-r1-p1-4x8-arch-1006-1347-8485-7-afb90`; Succeeded; {'active': 0, 'succeeded': 4, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 14300, 'eta': '1m', 'grad': 0.0066, 'ce': 2.504}.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_135053-olmo3-1b-ug-partial-p2qk-s480-p80-linear-qk-silu-down-rms-fp8-30b', '/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_135053-olmo3-1b-ug-partial-p2qk-s480-p80-linear-qk-silu-down-rms-fp8-30b/logs'].
- p1: `mose-recover-p1-r3-p1-1x8-pool-1006-1358-5ee-ff2d5`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 12430, 'eta': '1h34m', 'grad': 0.0144, 'ce': 2.576}.
- p0qk: `mose-ug-p0qk-r1-p1-1x8-pool-1006-1347-459e-3-74da9`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 12490, 'eta': '1h31m', 'grad': 0.0168, 'ce': 2.526}.
- p2: `mose-recover-p2-r3-p1-2x8-arch-1006-1409-7a8-a7469`; Succeeded; {'active': 0, 'succeeded': 2, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 14300, 'eta': '1m', 'grad': 0.0068, 'ce': 2.508}.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_141212-olmo3-1b-ug-partial-p2-s480-p80-linear-down-rms-fp8-30b', '/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_141212-olmo3-1b-ug-partial-p2-s480-p80-linear-down-rms-fp8-30b/logs'].
- p3qk: `mose-ug-p3qk-r1-p1-1x8-pool-1006-2111-2dcc-7-ec6b7`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 3540, 'eta': '9h39m', 'grad': 0.0318, 'ce': 2.842}.
- p4qk: `mose-ug-p4qk-r1-p1-1x8-pool-1006-2111-0416-8-5686b`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 3530, 'eta': '9h40m', 'grad': 0.064, 'ce': 2.763}.
- f1: `mose-recover-f1-r2-p1-1x8-pool-1006-1207-08d-af98d`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 14300, 'eta': '1m', 'grad': 0.008, 'ce': 2.505}.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_121016-olmo3-1b-ffn-budget-f1-ug384-down768-unshared-linear-rms-fp8-30b', '/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_121016-olmo3-1b-ffn-budget-f1-ug384-down768-unshared-linear-rms-fp8-30b/logs'].
- f2: `mose-recover-f2-r2-p1-1x8-pool-1006-1213-0e1-bdefa`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 14300, 'eta': '1m', 'grad': 0.0072, 'ce': 2.521}.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_121541-olmo3-1b-ffn-budget-f2-ug256-down1024-unshared-linear-rms-fp8-30b', '/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_121541-olmo3-1b-ffn-budget-f2-ug256-down1024-unshared-linear-rms-fp8-30b/logs'].
- f3: `mose-recover-f3-r2-p1-1x8-pool-1006-1213-ccc-5d862`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 14300, 'eta': '1m', 'grad': 0.0083, 'ce': 2.504}.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_121614-olmo3-1b-ffn-budget-f3-ug480-down672-shared-linear-rms-fp8-30b', '/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_121614-olmo3-1b-ffn-budget-f3-ug480-down672-shared-linear-rms-fp8-30b/logs'].
- f3qk: `mose-ffn-f3qk-r1-p1-4x8-arch-1006-1923-76a7-5350e`; Succeeded; {'active': 0, 'succeeded': 4, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 14300, 'eta': '1m', 'grad': 0.0071, 'ce': 2.5}.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_192653-olmo3-1b-ffn-budget-f3qk-ug480-down672-shared-qk-silu-ug-linear-down-rms-fp8-30b', '/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_192653-olmo3-1b-ffn-budget-f3qk-ug480-down672-shared-qk-silu-ug-linear-down-rms-fp8-30b/logs'].
- f2qk: `mose-ffn-f2qk-r1-p1-1x8-pool-1006-1922-88d3-ed4c5`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 5080, 'eta': '7h46m', 'grad': 0.0506, 'ce': 2.759}.
- f1qk: `mose-ffn-f1qk-r2-p1-1x8-arch-1006-1937-bfde-3c52a`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 5530, 'eta': '7h33m', 'grad': 0.0763, 'ce': 2.697}.
- f4qk: `mose-ffn-f4qk-r1-p1-2x8-arch-1006-2300-6d41-46be5`; Running; {'active': 2, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 3090, 'eta': '4h50m', 'grad': 0.0448, 'ce': 2.8}.
- f5qk: `mose-ffn-f5qk-r1-p1-2x8-arch-1006-2300-f278-d8613`; Running; {'active': 2, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 3140, 'eta': '4h43m', 'grad': 0.0298, 'ce': 2.768}.
- c0s2: `mose-r512-c0s2-r1-p1-1x8-pool-1006-2106-ea9c-f2516`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 3830, 'eta': '8h53m', 'grad': 0.0309, 'ce': 2.734}.
- c6s2: `mose-r512-c6s2-r1-p1-1x8-pool-1006-2107-3d78-52e71`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 3710, 'eta': '9h16m', 'grad': 0.052, 'ce': 2.789}.
- c0s3: `mose-r512-c0s3-r1-p1-1x8-pool-1006-2107-2f04-80461`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 3800, 'eta': '8h57m', 'grad': 0.0468, 'ce': 2.82}.
- c6s3: `mose-r512-c6s3-r1-p1-1x8-pool-1006-2108-31a7-fc86e`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 3760, 'eta': '9h4m', 'grad': 0.0518, 'ce': 2.764}.
- c2s2: `mose-r512-c2s2-r1-p1-1x8-pool-1006-2338-833a-1f47a`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 850, 'eta': '11h25m', 'grad': 0.1997, 'ce': 3.241}.
- c2s3: `mose-r512-c2s3-r1-p1-1x8-pool-1006-2338-d226-b5503`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 810, 'eta': '11h45m', 'grad': 0.2166, 'ce': 3.256}.

## 2026-10-07 00:56 HKT — monitoring

- p0: `mose-ug-p0-r1-p1-1x8-pool-1006-1123-7104-77485485`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 14300, 'eta': '1m', 'grad': 0.0075, 'ce': 2.503}.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_112904-olmo3-1b-ug-partial-p0-s0-p512-linear-down-rms-fp8-30b', '/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_112904-olmo3-1b-ug-partial-p0-s0-p512-linear-down-rms-fp8-30b/logs'].
- p4: `mose-recover-p4-r2-p1-1x8-pool-1006-1207-0d3-44098`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 14120, 'eta': '11m', 'grad': 0.0086, 'ce': 2.456}.
- p3: `mose-recover-p3-r3-p1-1x8-pool-1006-1228-c38-15484`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 13940, 'eta': '20m', 'grad': 0.0162, 'ce': 2.464}.
- p2qk: `mose-ug-p2qk-r1-p1-4x8-arch-1006-1347-8485-7-afb90`; Succeeded; {'active': 0, 'succeeded': 4, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 14300, 'eta': '1m', 'grad': 0.0066, 'ce': 2.504}.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_135053-olmo3-1b-ug-partial-p2qk-s480-p80-linear-qk-silu-down-rms-fp8-30b', '/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_135053-olmo3-1b-ug-partial-p2qk-s480-p80-linear-qk-silu-down-rms-fp8-30b/logs'].
- p1: `mose-recover-p1-r3-p1-1x8-pool-1006-1358-5ee-ff2d5`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 12990, 'eta': '1h6m', 'grad': 0.016, 'ce': 2.508}.
- p0qk: `mose-ug-p0qk-r1-p1-1x8-pool-1006-1347-459e-3-74da9`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 13060, 'eta': '1h3m', 'grad': 0.0147, 'ce': 2.493}.
- p2: `mose-recover-p2-r3-p1-2x8-arch-1006-1409-7a8-a7469`; Succeeded; {'active': 0, 'succeeded': 2, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 14300, 'eta': '1m', 'grad': 0.0068, 'ce': 2.508}.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_141212-olmo3-1b-ug-partial-p2-s480-p80-linear-down-rms-fp8-30b', '/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_141212-olmo3-1b-ug-partial-p2-s480-p80-linear-down-rms-fp8-30b/logs'].
- p3qk: `mose-ug-p3qk-r1-p1-1x8-pool-1006-2111-2dcc-7-ec6b7`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 4110, 'eta': '9h7m', 'grad': 0.0379, 'ce': 2.7}.
- p4qk: `mose-ug-p4qk-r1-p1-1x8-pool-1006-2111-0416-8-5686b`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 4090, 'eta': '9h10m', 'grad': 0.0416, 'ce': 2.745}.
- f1: `mose-recover-f1-r2-p1-1x8-pool-1006-1207-08d-af98d`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 14300, 'eta': '1m', 'grad': 0.008, 'ce': 2.505}.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_121016-olmo3-1b-ffn-budget-f1-ug384-down768-unshared-linear-rms-fp8-30b', '/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_121016-olmo3-1b-ffn-budget-f1-ug384-down768-unshared-linear-rms-fp8-30b/logs'].
- f2: `mose-recover-f2-r2-p1-1x8-pool-1006-1213-0e1-bdefa`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 14300, 'eta': '1m', 'grad': 0.0072, 'ce': 2.521}.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_121541-olmo3-1b-ffn-budget-f2-ug256-down1024-unshared-linear-rms-fp8-30b', '/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_121541-olmo3-1b-ffn-budget-f2-ug256-down1024-unshared-linear-rms-fp8-30b/logs'].
- f3: `mose-recover-f3-r2-p1-1x8-pool-1006-1213-ccc-5d862`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 14300, 'eta': '1m', 'grad': 0.0083, 'ce': 2.504}.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_121614-olmo3-1b-ffn-budget-f3-ug480-down672-shared-linear-rms-fp8-30b', '/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_121614-olmo3-1b-ffn-budget-f3-ug480-down672-shared-linear-rms-fp8-30b/logs'].
- f3qk: `mose-ffn-f3qk-r1-p1-4x8-arch-1006-1923-76a7-5350e`; Succeeded; {'active': 0, 'succeeded': 4, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 14300, 'eta': '1m', 'grad': 0.0071, 'ce': 2.5}.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_192653-olmo3-1b-ffn-budget-f3qk-ug480-down672-shared-qk-silu-ug-linear-down-rms-fp8-30b', '/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_192653-olmo3-1b-ffn-budget-f3qk-ug480-down672-shared-qk-silu-ug-linear-down-rms-fp8-30b/logs'].
- f2qk: `mose-ffn-f2qk-r1-p1-1x8-pool-1006-1922-88d3-ed4c5`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 5680, 'eta': '7h15m', 'grad': 0.0274, 'ce': 2.74}.
- f1qk: `mose-ffn-f1qk-r2-p1-1x8-arch-1006-1937-bfde-3c52a`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 6110, 'eta': '7h3m', 'grad': 0.058, 'ce': 2.624}.
- f4qk: `mose-ffn-f4qk-r1-p1-2x8-arch-1006-2300-6d41-46be5`; Running; {'active': 2, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 4260, 'eta': '4h20m', 'grad': 0.0237, 'ce': 2.795}.
- f5qk: `mose-ffn-f5qk-r1-p1-2x8-arch-1006-2300-f278-d8613`; Running; {'active': 2, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 4330, 'eta': '4h12m', 'grad': 0.0326, 'ce': 2.673}.
- c0s2: `mose-r512-c0s2-r1-p1-1x8-pool-1006-2106-ea9c-f2516`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 4420, 'eta': '8h23m', 'grad': 0.0213, 'ce': 2.713}.
- c6s2: `mose-r512-c6s2-r1-p1-1x8-pool-1006-2107-3d78-52e71`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 4290, 'eta': '8h45m', 'grad': 0.0315, 'ce': 2.711}.
- c0s3: `mose-r512-c0s3-r1-p1-1x8-pool-1006-2107-2f04-80461`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 4390, 'eta': '8h26m', 'grad': 0.0524, 'ce': 2.723}.
- c6s3: `mose-r512-c6s3-r1-p1-1x8-pool-1006-2108-31a7-fc86e`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 4350, 'eta': '8h33m', 'grad': 0.0691, 'ce': 2.681}.
- c2s2: `mose-r512-c2s2-r1-p1-1x8-pool-1006-2338-833a-1f47a`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 1440, 'eta': '10h57m', 'grad': 0.1825, 'ce': 3.043}.
- c2s3: `mose-r512-c2s3-r1-p1-1x8-pool-1006-2338-d226-b5503`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 1380, 'eta': '11h17m', 'grad': 0.1707, 'ce': 3.119}.

## 2026-10-07 01:26 HKT — monitoring

- p0: `mose-ug-p0-r1-p1-1x8-pool-1006-1123-7104-77485485`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 14300, 'eta': '1m', 'grad': 0.0075, 'ce': 2.503}.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_112904-olmo3-1b-ug-partial-p0-s0-p512-linear-down-rms-fp8-30b', '/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_112904-olmo3-1b-ug-partial-p0-s0-p512-linear-down-rms-fp8-30b/logs'].
- p4: `mose-recover-p4-r2-p1-1x8-pool-1006-1207-0d3-44098`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 14300, 'eta': '1m', 'grad': 0.0093, 'ce': 2.508}.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_120940-olmo3-1b-ug-partial-p4-s160-p368-linear-down-rms-fp8-30b', '/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_120940-olmo3-1b-ug-partial-p4-s160-p368-linear-down-rms-fp8-30b/logs'].
- p3: `mose-recover-p3-r3-p1-1x8-pool-1006-1228-c38-15484`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 14300, 'eta': '1m', 'grad': 0.0115, 'ce': 2.507}.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_123139-olmo3-1b-ug-partial-p3-s320-p224-linear-down-rms-fp8-30b', '/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_123139-olmo3-1b-ug-partial-p3-s320-p224-linear-down-rms-fp8-30b/logs'].
- p2qk: `mose-ug-p2qk-r1-p1-4x8-arch-1006-1347-8485-7-afb90`; Succeeded; {'active': 0, 'succeeded': 4, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 14300, 'eta': '1m', 'grad': 0.0066, 'ce': 2.504}.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_135053-olmo3-1b-ug-partial-p2qk-s480-p80-linear-qk-silu-down-rms-fp8-30b', '/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_135053-olmo3-1b-ug-partial-p2qk-s480-p80-linear-qk-silu-down-rms-fp8-30b/logs'].
- p1: `mose-recover-p1-r3-p1-1x8-pool-1006-1358-5ee-ff2d5`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 13590, 'eta': '36m', 'grad': 0.009, 'ce': 2.469}.
- p0qk: `mose-ug-p0qk-r1-p1-1x8-pool-1006-1347-459e-3-74da9`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 13660, 'eta': '33m', 'grad': 0.0114, 'ce': 2.435}.
- p2: `mose-recover-p2-r3-p1-2x8-arch-1006-1409-7a8-a7469`; Succeeded; {'active': 0, 'succeeded': 2, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 14300, 'eta': '1m', 'grad': 0.0068, 'ce': 2.508}.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_141212-olmo3-1b-ug-partial-p2-s480-p80-linear-down-rms-fp8-30b', '/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_141212-olmo3-1b-ug-partial-p2-s480-p80-linear-down-rms-fp8-30b/logs'].
- p3qk: `mose-ug-p3qk-r1-p1-1x8-pool-1006-2111-2dcc-7-ec6b7`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 4670, 'eta': '8h37m', 'grad': 0.0432, 'ce': 2.708}.
- p4qk: `mose-ug-p4qk-r1-p1-1x8-pool-1006-2111-0416-8-5686b`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 4650, 'eta': '8h39m', 'grad': 0.0502, 'ce': 2.663}.
- f1: `mose-recover-f1-r2-p1-1x8-pool-1006-1207-08d-af98d`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 14300, 'eta': '1m', 'grad': 0.008, 'ce': 2.505}.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_121016-olmo3-1b-ffn-budget-f1-ug384-down768-unshared-linear-rms-fp8-30b', '/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_121016-olmo3-1b-ffn-budget-f1-ug384-down768-unshared-linear-rms-fp8-30b/logs'].
- f2: `mose-recover-f2-r2-p1-1x8-pool-1006-1213-0e1-bdefa`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 14300, 'eta': '1m', 'grad': 0.0072, 'ce': 2.521}.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_121541-olmo3-1b-ffn-budget-f2-ug256-down1024-unshared-linear-rms-fp8-30b', '/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_121541-olmo3-1b-ffn-budget-f2-ug256-down1024-unshared-linear-rms-fp8-30b/logs'].
- f3: `mose-recover-f3-r2-p1-1x8-pool-1006-1213-ccc-5d862`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 14300, 'eta': '1m', 'grad': 0.0083, 'ce': 2.504}.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_121614-olmo3-1b-ffn-budget-f3-ug480-down672-shared-linear-rms-fp8-30b', '/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_121614-olmo3-1b-ffn-budget-f3-ug480-down672-shared-linear-rms-fp8-30b/logs'].
- f3qk: `mose-ffn-f3qk-r1-p1-4x8-arch-1006-1923-76a7-5350e`; Succeeded; {'active': 0, 'succeeded': 4, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 14300, 'eta': '1m', 'grad': 0.0071, 'ce': 2.5}.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_192653-olmo3-1b-ffn-budget-f3qk-ug480-down672-shared-qk-silu-ug-linear-down-rms-fp8-30b', '/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_192653-olmo3-1b-ffn-budget-f3qk-ug480-down672-shared-qk-silu-ug-linear-down-rms-fp8-30b/logs'].
- f2qk: `mose-ffn-f2qk-r1-p1-1x8-pool-1006-1922-88d3-ed4c5`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 6280, 'eta': '6h45m', 'grad': 0.0257, 'ce': 2.703}.
- f1qk: `mose-ffn-f1qk-r2-p1-1x8-arch-1006-1937-bfde-3c52a`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 6700, 'eta': '6h32m', 'grad': 0.0864, 'ce': 2.582}.
- f4qk: `mose-ffn-f4qk-r1-p1-2x8-arch-1006-2300-6d41-46be5`; Running; {'active': 2, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 5430, 'eta': '3h49m', 'grad': 0.0181, 'ce': 2.7}.
- f5qk: `mose-ffn-f5qk-r1-p1-2x8-arch-1006-2300-f278-d8613`; Running; {'active': 2, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 5510, 'eta': '3h43m', 'grad': 0.0175, 'ce': 2.713}.
- c0s2: `mose-r512-c0s2-r1-p1-1x8-pool-1006-2106-ea9c-f2516`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 4990, 'eta': '7h53m', 'grad': 0.0425, 'ce': 2.746}.
- c6s2: `mose-r512-c6s2-r1-p1-1x8-pool-1006-2107-3d78-52e71`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 4870, 'eta': '8h14m', 'grad': 0.0216, 'ce': 2.747}.
- c0s3: `mose-r512-c0s3-r1-p1-1x8-pool-1006-2107-2f04-80461`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 4980, 'eta': '7h56m', 'grad': 0.0247, 'ce': 2.682}.
- c6s3: `mose-r512-c6s3-r1-p1-1x8-pool-1006-2108-31a7-fc86e`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 4940, 'eta': '8h2m', 'grad': 0.0314, 'ce': 2.695}.
- c2s2: `mose-r512-c2s2-r1-p1-1x8-pool-1006-2338-833a-1f47a`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 2000, 'eta': '10h36m', 'grad': 0.071, 'ce': 2.995}.
- c2s3: `mose-r512-c2s3-r1-p1-1x8-pool-1006-2338-d226-b5503`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 1950, 'eta': '10h49m', 'grad': 0.1185, 'ce': 3.023}.

## 2026-10-07 02:12 HKT — monitoring

- p0: `mose-ug-p0-r1-p1-1x8-pool-1006-1123-7104-77485485`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 14300, 'eta': '1m', 'grad': 0.0075, 'ce': 2.503}.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_112904-olmo3-1b-ug-partial-p0-s0-p512-linear-down-rms-fp8-30b', '/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_112904-olmo3-1b-ug-partial-p0-s0-p512-linear-down-rms-fp8-30b/logs'].
- p4: `mose-recover-p4-r2-p1-1x8-pool-1006-1207-0d3-44098`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 14300, 'eta': '1m', 'grad': 0.0093, 'ce': 2.508}.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_120940-olmo3-1b-ug-partial-p4-s160-p368-linear-down-rms-fp8-30b', '/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_120940-olmo3-1b-ug-partial-p4-s160-p368-linear-down-rms-fp8-30b/logs'].
- p3: `mose-recover-p3-r3-p1-1x8-pool-1006-1228-c38-15484`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 14300, 'eta': '1m', 'grad': 0.0115, 'ce': 2.507}.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_123139-olmo3-1b-ug-partial-p3-s320-p224-linear-down-rms-fp8-30b', '/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_123139-olmo3-1b-ug-partial-p3-s320-p224-linear-down-rms-fp8-30b/logs'].
- p2qk: `mose-ug-p2qk-r1-p1-4x8-arch-1006-1347-8485-7-afb90`; Succeeded; {'active': 0, 'succeeded': 4, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 14300, 'eta': '1m', 'grad': 0.0066, 'ce': 2.504}.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_135053-olmo3-1b-ug-partial-p2qk-s480-p80-linear-qk-silu-down-rms-fp8-30b', '/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_135053-olmo3-1b-ug-partial-p2qk-s480-p80-linear-qk-silu-down-rms-fp8-30b/logs'].
- p1: `mose-recover-p1-r3-p1-1x8-pool-1006-1358-5ee-ff2d5`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 14200, 'eta': '6m', 'grad': 0.0079, 'ce': 2.516}.
- p0qk: `mose-ug-p0qk-r1-p1-1x8-pool-1006-1347-459e-3-74da9`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 14270, 'eta': '2m', 'grad': 0.008, 'ce': 2.569}.
- p2: `mose-recover-p2-r3-p1-2x8-arch-1006-1409-7a8-a7469`; Succeeded; {'active': 0, 'succeeded': 2, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 14300, 'eta': '1m', 'grad': 0.0068, 'ce': 2.508}.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_141212-olmo3-1b-ug-partial-p2-s480-p80-linear-down-rms-fp8-30b', '/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_141212-olmo3-1b-ug-partial-p2-s480-p80-linear-down-rms-fp8-30b/logs'].
- p3qk: `mose-ug-p3qk-r1-p1-1x8-pool-1006-2111-2dcc-7-ec6b7`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 5240, 'eta': '8h6m', 'grad': 0.0445, 'ce': 2.694}.
- p4qk: `mose-ug-p4qk-r1-p1-1x8-pool-1006-2111-0416-8-5686b`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 5220, 'eta': '8h8m', 'grad': 0.025, 'ce': 2.715}.
- f1: `mose-recover-f1-r2-p1-1x8-pool-1006-1207-08d-af98d`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 14300, 'eta': '1m', 'grad': 0.008, 'ce': 2.505}.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_121016-olmo3-1b-ffn-budget-f1-ug384-down768-unshared-linear-rms-fp8-30b', '/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_121016-olmo3-1b-ffn-budget-f1-ug384-down768-unshared-linear-rms-fp8-30b/logs'].
- f2: `mose-recover-f2-r2-p1-1x8-pool-1006-1213-0e1-bdefa`; UNKNOWN; None; None.
- f3: `mose-recover-f3-r2-p1-1x8-pool-1006-1213-ccc-5d862`; UNKNOWN; None; None.
- f3qk: `mose-ffn-f3qk-r1-p1-4x8-arch-1006-1923-76a7-5350e`; Succeeded; {'active': 0, 'succeeded': 4, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 14300, 'eta': '1m', 'grad': 0.0071, 'ce': 2.5}.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_192653-olmo3-1b-ffn-budget-f3qk-ug480-down672-shared-qk-silu-ug-linear-down-rms-fp8-30b', '/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_192653-olmo3-1b-ffn-budget-f3qk-ug480-down672-shared-qk-silu-ug-linear-down-rms-fp8-30b/logs'].
- f2qk: `mose-ffn-f2qk-r1-p1-1x8-pool-1006-1922-88d3-ed4c5`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 7190, 'eta': '5h59m', 'grad': 0.0256, 'ce': 2.648}.
- f1qk: `mose-ffn-f1qk-r2-p1-1x8-arch-1006-1937-bfde-3c52a`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 7600, 'eta': '5h46m', 'grad': 0.0673, 'ce': 2.765}.
- f4qk: `mose-ffn-f4qk-r1-p1-2x8-arch-1006-2300-6d41-46be5`; Running; {'active': 2, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 7210, 'eta': '3h3m', 'grad': 0.0454, 'ce': 2.652}.
- f5qk: `mose-ffn-f5qk-r1-p1-2x8-arch-1006-2300-f278-d8613`; Running; {'active': 2, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 7350, 'eta': '2h56m', 'grad': 0.0162, 'ce': 2.693}.
- a1: `mose-attn-a1-r1-p1-1x8-pool-1007-0128-5292-1-78ab4`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 780, 'eta': '11h31m', 'grad': 0.177, 'ce': 3.28}.
- a2: `mose-attn-a2-r1-p1-1x8-pool-1007-0128-b35f-2-1cf16`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 780, 'eta': '11h29m', 'grad': 0.1802, 'ce': 3.282}.
- c0s2: `mose-r512-c0s2-r1-p1-1x8-pool-1006-2106-ea9c-f2516`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 5920, 'eta': '7h6m', 'grad': 0.0271, 'ce': 2.651}.
- c6s2: `mose-r512-c6s2-r1-p1-1x8-pool-1006-2107-3d78-52e71`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 5750, 'eta': '7h28m', 'grad': 0.0199, 'ce': 2.634}.
- c0s3: `mose-r512-c0s3-r1-p1-1x8-pool-1006-2107-2f04-80461`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 5880, 'eta': '7h10m', 'grad': 0.0393, 'ce': 2.661}.
- c6s3: `mose-r512-c6s3-r1-p1-1x8-pool-1006-2108-31a7-fc86e`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 5830, 'eta': '7h17m', 'grad': 0.0186, 'ce': 2.655}.
- c2s2: `mose-r512-c2s2-r1-p1-1x8-pool-1006-2338-833a-1f47a`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 2930, 'eta': '9h42m', 'grad': 0.018, 'ce': 2.826}.
- c2s3: `mose-r512-c2s3-r1-p1-1x8-pool-1006-2338-d226-b5503`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 2820, 'eta': '10h4m', 'grad': 0.0314, 'ce': 2.839}.

## 2026-10-07 02:26 HKT — monitoring

- p0: `mose-ug-p0-r1-p1-1x8-pool-1006-1123-7104-77485485`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_112904-olmo3-1b-ug-partial-p0-s0-p512-linear-down-rms-fp8-30b'].
- p4: `mose-recover-p4-r2-p1-1x8-pool-1006-1207-0d3-44098`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_120940-olmo3-1b-ug-partial-p4-s160-p368-linear-down-rms-fp8-30b'].
- p3: `mose-recover-p3-r3-p1-1x8-pool-1006-1228-c38-15484`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_123139-olmo3-1b-ug-partial-p3-s320-p224-linear-down-rms-fp8-30b'].
- p2qk: `mose-ug-p2qk-r1-p1-4x8-arch-1006-1347-8485-7-afb90`; Succeeded; {'active': 0, 'succeeded': 4, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_135053-olmo3-1b-ug-partial-p2qk-s480-p80-linear-qk-silu-down-rms-fp8-30b'].
- p1: `mose-recover-p1-r3-p1-1x8-pool-1006-1358-5ee-ff2d5`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 14300, 'eta': '1m', 'grad': 0.0118, 'ce': 2.506}.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_140134-olmo3-1b-ug-partial-p1-s560-p0-linear-down-rms-fp8-30b', '/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_140134-olmo3-1b-ug-partial-p1-s560-p0-linear-down-rms-fp8-30b/logs'].
- p0qk: `mose-ug-p0qk-r1-p1-1x8-pool-1006-1347-459e-3-74da9`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 14300, 'eta': '1m', 'grad': 0.0073, 'ce': 2.501}.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_140141-olmo3-1b-ug-partial-p0qk-s0-p512-linear-qk-silu-down-rms-fp8-30b', '/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_140141-olmo3-1b-ug-partial-p0qk-s0-p512-linear-qk-silu-down-rms-fp8-30b/logs'].
- p2: `mose-recover-p2-r3-p1-2x8-arch-1006-1409-7a8-a7469`; Succeeded; {'active': 0, 'succeeded': 2, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_141212-olmo3-1b-ug-partial-p2-s480-p80-linear-down-rms-fp8-30b'].
- p3qk: `mose-ug-p3qk-r1-p1-1x8-pool-1006-2111-2dcc-7-ec6b7`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 5800, 'eta': '7h36m', 'grad': 0.0259, 'ce': 2.697}.
- p4qk: `mose-ug-p4qk-r1-p1-1x8-pool-1006-2111-0416-8-5686b`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 5780, 'eta': '7h38m', 'grad': 0.0213, 'ce': 2.635}.
- f1: `mose-recover-f1-r2-p1-1x8-pool-1006-1207-08d-af98d`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_121016-olmo3-1b-ffn-budget-f1-ug384-down768-unshared-linear-rms-fp8-30b'].
- f2: `mose-recover-f2-r2-p1-1x8-pool-1006-1213-0e1-bdefa`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_121541-olmo3-1b-ffn-budget-f2-ug256-down1024-unshared-linear-rms-fp8-30b'].
- f3: `mose-recover-f3-r2-p1-1x8-pool-1006-1213-ccc-5d862`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_121614-olmo3-1b-ffn-budget-f3-ug480-down672-shared-linear-rms-fp8-30b'].
- f3qk: `mose-ffn-f3qk-r1-p1-4x8-arch-1006-1923-76a7-5350e`; Succeeded; {'active': 0, 'succeeded': 4, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_192653-olmo3-1b-ffn-budget-f3qk-ug480-down672-shared-qk-silu-ug-linear-down-rms-fp8-30b'].
- f2qk: `mose-ffn-f2qk-r1-p1-1x8-pool-1006-1922-88d3-ed4c5`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 7470, 'eta': '5h44m', 'grad': 0.0152, 'ce': 2.701}.
- f1qk: `mose-ffn-f1qk-r2-p1-1x8-arch-1006-1937-bfde-3c52a`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 7870, 'eta': '5h31m', 'grad': 0.0198, 'ce': 2.682}.
- f4qk: `mose-ffn-f4qk-r1-p1-2x8-arch-1006-2300-6d41-46be5`; Running; {'active': 2, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 7740, 'eta': '2h50m', 'grad': 0.0162, 'ce': 2.677}.
- f5qk: `mose-ffn-f5qk-r1-p1-2x8-arch-1006-2300-f278-d8613`; Running; {'active': 2, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 7900, 'eta': '2h42m', 'grad': 0.0386, 'ce': 2.623}.
- a1: `mose-attn-a1-r1-p1-1x8-pool-1007-0128-5292-1-78ab4`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 1020, 'eta': '11h29m', 'grad': 0.1097, 'ce': 3.189}.
- a2: `mose-attn-a2-r1-p1-1x8-pool-1007-0128-b35f-2-1cf16`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 1030, 'eta': '11h26m', 'grad': 0.1998, 'ce': 3.226}.
- c0s2: `mose-r512-c0s2-r1-p1-1x8-pool-1006-2106-ea9c-f2516`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 6180, 'eta': '6h54m', 'grad': 0.0335, 'ce': 2.694}.
- c6s2: `mose-r512-c6s2-r1-p1-1x8-pool-1006-2107-3d78-52e71`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 6000, 'eta': '7h16m', 'grad': 0.0205, 'ce': 2.65}.
- c0s3: `mose-r512-c0s3-r1-p1-1x8-pool-1006-2107-2f04-80461`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 6140, 'eta': '6h58m', 'grad': 0.0297, 'ce': 2.655}.
- c6s3: `mose-r512-c6s3-r1-p1-1x8-pool-1006-2108-31a7-fc86e`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 6080, 'eta': '7h4m', 'grad': 0.0186, 'ce': 2.65}.
- c2s2: `mose-r512-c2s2-r1-p1-1x8-pool-1006-2338-833a-1f47a`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 3180, 'eta': '9h32m', 'grad': 0.027, 'ce': 2.76}.
- c2s3: `mose-r512-c2s3-r1-p1-1x8-pool-1006-2338-d226-b5503`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 3070, 'eta': '9h53m', 'grad': 0.0564, 'ce': 2.83}.

## 2026-10-07 02:29 HKT — originalten and unified reference comparison complete

P1/P0qk final14306 checkpoints and Succeeded/active0 verified, exact-segment uploads Finished syncing confirmed. All originalten plusF3qk complete;14of25 training configs remain active. Analysis `mose-report-original-batch-r1-p1-1x8-pool-10-524a5` Succeeded/active0; two fullJSON SHA256 verified. P0qk2.477410089 andF3qk2.477421343 effectivelytied, both worse thanC6reference2.474762967. See ORIGINAL_BATCH_RESULTS.md.

## 2026-10-07 02:56 HKT — monitoring

- p0: `mose-ug-p0-r1-p1-1x8-pool-1006-1123-7104-77485485`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_112904-olmo3-1b-ug-partial-p0-s0-p512-linear-down-rms-fp8-30b'].
- p4: `mose-recover-p4-r2-p1-1x8-pool-1006-1207-0d3-44098`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_120940-olmo3-1b-ug-partial-p4-s160-p368-linear-down-rms-fp8-30b'].
- p3: `mose-recover-p3-r3-p1-1x8-pool-1006-1228-c38-15484`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_123139-olmo3-1b-ug-partial-p3-s320-p224-linear-down-rms-fp8-30b'].
- p2qk: `mose-ug-p2qk-r1-p1-4x8-arch-1006-1347-8485-7-afb90`; Succeeded; {'active': 0, 'succeeded': 4, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_135053-olmo3-1b-ug-partial-p2qk-s480-p80-linear-qk-silu-down-rms-fp8-30b'].
- p1: `mose-recover-p1-r3-p1-1x8-pool-1006-1358-5ee-ff2d5`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_140134-olmo3-1b-ug-partial-p1-s560-p0-linear-down-rms-fp8-30b'].
- p0qk: `mose-ug-p0qk-r1-p1-1x8-pool-1006-1347-459e-3-74da9`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_140141-olmo3-1b-ug-partial-p0qk-s0-p512-linear-qk-silu-down-rms-fp8-30b'].
- p2: `mose-recover-p2-r3-p1-2x8-arch-1006-1409-7a8-a7469`; Succeeded; {'active': 0, 'succeeded': 2, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_141212-olmo3-1b-ug-partial-p2-s480-p80-linear-down-rms-fp8-30b'].
- p3qk: `mose-ug-p3qk-r1-p1-1x8-pool-1006-2111-2dcc-7-ec6b7`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 6360, 'eta': '7h5m', 'grad': 0.0672, 'ce': 2.655}.
- p4qk: `mose-ug-p4qk-r1-p1-1x8-pool-1006-2111-0416-8-5686b`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 6340, 'eta': '7h8m', 'grad': 0.0197, 'ce': 2.67}.
- f1: `mose-recover-f1-r2-p1-1x8-pool-1006-1207-08d-af98d`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_121016-olmo3-1b-ffn-budget-f1-ug384-down768-unshared-linear-rms-fp8-30b'].
- f2: `mose-recover-f2-r2-p1-1x8-pool-1006-1213-0e1-bdefa`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_121541-olmo3-1b-ffn-budget-f2-ug256-down1024-unshared-linear-rms-fp8-30b'].
- f3: `mose-recover-f3-r2-p1-1x8-pool-1006-1213-ccc-5d862`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_121614-olmo3-1b-ffn-budget-f3-ug480-down672-shared-linear-rms-fp8-30b'].
- f3qk: `mose-ffn-f3qk-r1-p1-4x8-arch-1006-1923-76a7-5350e`; Succeeded; {'active': 0, 'succeeded': 4, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_192653-olmo3-1b-ffn-budget-f3qk-ug480-down672-shared-qk-silu-ug-linear-down-rms-fp8-30b'].
- f2qk: `mose-ffn-f2qk-r1-p1-1x8-pool-1006-1922-88d3-ed4c5`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 8050, 'eta': '5h16m', 'grad': 0.022, 'ce': 2.556}.
- f1qk: `mose-ffn-f1qk-r2-p1-1x8-arch-1006-1937-bfde-3c52a`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 8460, 'eta': '5h1m', 'grad': 0.0574, 'ce': 2.685}.
- f4qk: `mose-ffn-f4qk-r1-p1-2x8-arch-1006-2300-6d41-46be5`; Running; {'active': 2, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 8910, 'eta': '2h19m', 'grad': 0.0435, 'ce': 2.582}.
- f5qk: `mose-ffn-f5qk-r1-p1-2x8-arch-1006-2300-f278-d8613`; Running; {'active': 2, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 9070, 'eta': '2h13m', 'grad': 0.0223, 'ce': 2.604}.
- a1: `mose-attn-a1-r1-p1-1x8-pool-1007-0128-5292-1-78ab4`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 1610, 'eta': '10h55m', 'grad': 0.1347, 'ce': 3.029}.
- a2: `mose-attn-a2-r1-p1-1x8-pool-1007-0128-b35f-2-1cf16`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 1620, 'eta': '10h52m', 'grad': 0.1166, 'ce': 3.076}.
- c0s2: `mose-r512-c0s2-r1-p1-1x8-pool-1006-2106-ea9c-f2516`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 6770, 'eta': '6h24m', 'grad': 0.0256, 'ce': 2.646}.
- c6s2: `mose-r512-c6s2-r1-p1-1x8-pool-1006-2107-3d78-52e71`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 6570, 'eta': '6h45m', 'grad': 0.0296, 'ce': 2.697}.
- c0s3: `mose-r512-c0s3-r1-p1-1x8-pool-1006-2107-2f04-80461`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 6730, 'eta': '6h27m', 'grad': 0.0212, 'ce': 2.687}.
- c6s3: `mose-r512-c6s3-r1-p1-1x8-pool-1006-2108-31a7-fc86e`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 6670, 'eta': '6h34m', 'grad': 0.0198, 'ce': 2.675}.
- c2s2: `mose-r512-c2s2-r1-p1-1x8-pool-1006-2338-833a-1f47a`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 3770, 'eta': '9h', 'grad': 0.0173, 'ce': 2.732}.
- c2s3: `mose-r512-c2s3-r1-p1-1x8-pool-1006-2338-d226-b5503`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 3640, 'eta': '9h22m', 'grad': 0.044, 'ce': 2.764}.

## 2026-10-07 03:26 HKT — monitoring

- p0: `mose-ug-p0-r1-p1-1x8-pool-1006-1123-7104-77485485`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_112904-olmo3-1b-ug-partial-p0-s0-p512-linear-down-rms-fp8-30b'].
- p4: `mose-recover-p4-r2-p1-1x8-pool-1006-1207-0d3-44098`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_120940-olmo3-1b-ug-partial-p4-s160-p368-linear-down-rms-fp8-30b'].
- p3: `mose-recover-p3-r3-p1-1x8-pool-1006-1228-c38-15484`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_123139-olmo3-1b-ug-partial-p3-s320-p224-linear-down-rms-fp8-30b'].
- p2qk: `mose-ug-p2qk-r1-p1-4x8-arch-1006-1347-8485-7-afb90`; Succeeded; {'active': 0, 'succeeded': 4, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_135053-olmo3-1b-ug-partial-p2qk-s480-p80-linear-qk-silu-down-rms-fp8-30b'].
- p1: `mose-recover-p1-r3-p1-1x8-pool-1006-1358-5ee-ff2d5`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_140134-olmo3-1b-ug-partial-p1-s560-p0-linear-down-rms-fp8-30b'].
- p0qk: `mose-ug-p0qk-r1-p1-1x8-pool-1006-1347-459e-3-74da9`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_140141-olmo3-1b-ug-partial-p0qk-s0-p512-linear-qk-silu-down-rms-fp8-30b'].
- p2: `mose-recover-p2-r3-p1-2x8-arch-1006-1409-7a8-a7469`; Succeeded; {'active': 0, 'succeeded': 2, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_141212-olmo3-1b-ug-partial-p2-s480-p80-linear-down-rms-fp8-30b'].
- p3qk: `mose-ug-p3qk-r1-p1-1x8-pool-1006-2111-2dcc-7-ec6b7`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 6920, 'eta': '6h35m', 'grad': 0.0883, 'ce': 2.618}.
- p4qk: `mose-ug-p4qk-r1-p1-1x8-pool-1006-2111-0416-8-5686b`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 6900, 'eta': '6h37m', 'grad': 0.0209, 'ce': 2.662}.
- f1: `mose-recover-f1-r2-p1-1x8-pool-1006-1207-08d-af98d`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_121016-olmo3-1b-ffn-budget-f1-ug384-down768-unshared-linear-rms-fp8-30b'].
- f2: `mose-recover-f2-r2-p1-1x8-pool-1006-1213-0e1-bdefa`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_121541-olmo3-1b-ffn-budget-f2-ug256-down1024-unshared-linear-rms-fp8-30b'].
- f3: `mose-recover-f3-r2-p1-1x8-pool-1006-1213-ccc-5d862`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_121614-olmo3-1b-ffn-budget-f3-ug480-down672-shared-linear-rms-fp8-30b'].
- f3qk: `mose-ffn-f3qk-r1-p1-4x8-arch-1006-1923-76a7-5350e`; Succeeded; {'active': 0, 'succeeded': 4, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_192653-olmo3-1b-ffn-budget-f3qk-ug480-down672-shared-qk-silu-ug-linear-down-rms-fp8-30b'].
- f2qk: `mose-ffn-f2qk-r1-p1-1x8-pool-1006-1922-88d3-ed4c5`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 8650, 'eta': '4h45m', 'grad': 0.0233, 'ce': 2.631}.
- f1qk: `mose-ffn-f1qk-r2-p1-1x8-arch-1006-1937-bfde-3c52a`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 9020, 'eta': '4h33m', 'grad': 0.0601, 'ce': 2.604}.
- f4qk: `mose-ffn-f4qk-r1-p1-2x8-arch-1006-2300-6d41-46be5`; Running; {'active': 2, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 10060, 'eta': '1h50m', 'grad': 0.0133, 'ce': 2.609}.
- f5qk: `mose-ffn-f5qk-r1-p1-2x8-arch-1006-2300-f278-d8613`; Running; {'active': 2, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 10270, 'eta': '1h42m', 'grad': 0.0215, 'ce': 2.62}.
- f4: `mose-ffn-f4-r1-p1-1x8-pool-1007-0257-f50a-58255159`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 490, 'eta': '11h5m', 'grad': 0.2012, 'ce': 3.521}.
- f5: `mose-ffn-f5-r1-p1-1x8-pool-1007-0257-2198-70783152`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 490, 'eta': '11h6m', 'grad': 0.2478, 'ce': 3.523}.
- a1: `mose-attn-a1-r1-p1-1x8-pool-1007-0128-5292-1-78ab4`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 2200, 'eta': '10h24m', 'grad': 0.0649, 'ce': 2.957}.
- a2: `mose-attn-a2-r1-p1-1x8-pool-1007-0128-b35f-2-1cf16`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 2210, 'eta': '10h21m', 'grad': 0.0853, 'ce': 2.893}.
- c0s2: `mose-r512-c0s2-r1-p1-1x8-pool-1006-2106-ea9c-f2516`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 7370, 'eta': '5h53m', 'grad': 0.0256, 'ce': 2.674}.
- c6s2: `mose-r512-c6s2-r1-p1-1x8-pool-1006-2107-3d78-52e71`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 7150, 'eta': '6h15m', 'grad': 0.018, 'ce': 2.678}.
- c0s3: `mose-r512-c0s3-r1-p1-1x8-pool-1006-2107-2f04-80461`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 7320, 'eta': '5h57m', 'grad': 0.0184, 'ce': 2.715}.
- c6s3: `mose-r512-c6s3-r1-p1-1x8-pool-1006-2108-31a7-fc86e`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 7260, 'eta': '6h3m', 'grad': 0.0156, 'ce': 2.594}.
- c2s2: `mose-r512-c2s2-r1-p1-1x8-pool-1006-2338-833a-1f47a`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 4360, 'eta': '8h29m', 'grad': 0.0254, 'ce': 2.665}.
- c2s3: `mose-r512-c2s3-r1-p1-1x8-pool-1006-2338-d226-b5503`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 4220, 'eta': '8h51m', 'grad': 0.0198, 'ce': 2.759}.

## 2026-10-07 03:56 HKT — monitoring

- p0: `mose-ug-p0-r1-p1-1x8-pool-1006-1123-7104-77485485`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_112904-olmo3-1b-ug-partial-p0-s0-p512-linear-down-rms-fp8-30b'].
- p4: `mose-recover-p4-r2-p1-1x8-pool-1006-1207-0d3-44098`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_120940-olmo3-1b-ug-partial-p4-s160-p368-linear-down-rms-fp8-30b'].
- p3: `mose-recover-p3-r3-p1-1x8-pool-1006-1228-c38-15484`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_123139-olmo3-1b-ug-partial-p3-s320-p224-linear-down-rms-fp8-30b'].
- p2qk: `mose-ug-p2qk-r1-p1-4x8-arch-1006-1347-8485-7-afb90`; Succeeded; {'active': 0, 'succeeded': 4, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_135053-olmo3-1b-ug-partial-p2qk-s480-p80-linear-qk-silu-down-rms-fp8-30b'].
- p1: `mose-recover-p1-r3-p1-1x8-pool-1006-1358-5ee-ff2d5`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_140134-olmo3-1b-ug-partial-p1-s560-p0-linear-down-rms-fp8-30b'].
- p0qk: `mose-ug-p0qk-r1-p1-1x8-pool-1006-1347-459e-3-74da9`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_140141-olmo3-1b-ug-partial-p0qk-s0-p512-linear-qk-silu-down-rms-fp8-30b'].
- p2: `mose-recover-p2-r3-p1-2x8-arch-1006-1409-7a8-a7469`; Succeeded; {'active': 0, 'succeeded': 2, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_141212-olmo3-1b-ug-partial-p2-s480-p80-linear-down-rms-fp8-30b'].
- p3qk: `mose-ug-p3qk-r1-p1-1x8-pool-1006-2111-2dcc-7-ec6b7`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 7490, 'eta': '6h5m', 'grad': 0.069, 'ce': 2.633}.
- p4qk: `mose-ug-p4qk-r1-p1-1x8-pool-1006-2111-0416-8-5686b`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 7460, 'eta': '6h7m', 'grad': 0.018, 'ce': 2.624}.
- f1: `mose-recover-f1-r2-p1-1x8-pool-1006-1207-08d-af98d`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_121016-olmo3-1b-ffn-budget-f1-ug384-down768-unshared-linear-rms-fp8-30b'].
- f2: `mose-recover-f2-r2-p1-1x8-pool-1006-1213-0e1-bdefa`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_121541-olmo3-1b-ffn-budget-f2-ug256-down1024-unshared-linear-rms-fp8-30b'].
- f3: `mose-recover-f3-r2-p1-1x8-pool-1006-1213-ccc-5d862`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_121614-olmo3-1b-ffn-budget-f3-ug480-down672-shared-linear-rms-fp8-30b'].
- f3qk: `mose-ffn-f3qk-r1-p1-4x8-arch-1006-1923-76a7-5350e`; Succeeded; {'active': 0, 'succeeded': 4, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_192653-olmo3-1b-ffn-budget-f3qk-ug480-down672-shared-qk-silu-ug-linear-down-rms-fp8-30b'].
- f2qk: `mose-ffn-f2qk-r1-p1-1x8-pool-1006-1922-88d3-ed4c5`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 9250, 'eta': '4h15m', 'grad': 0.0246, 'ce': 2.68}.
- f1qk: `mose-ffn-f1qk-r2-p1-1x8-arch-1006-1937-bfde-3c52a`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 9610, 'eta': '4h2m', 'grad': 0.04, 'ce': 2.598}.
- f4qk: `mose-ffn-f4qk-r1-p1-2x8-arch-1006-2300-6d41-46be5`; Running; {'active': 2, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 11230, 'eta': '1h20m', 'grad': 0.0262, 'ce': 2.63}.
- f5qk: `mose-ffn-f5qk-r1-p1-2x8-arch-1006-2300-f278-d8613`; Running; {'active': 2, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 11470, 'eta': '1h12m', 'grad': 0.0174, 'ce': 2.544}.
- f4: `mose-ffn-f4-r1-p1-1x8-pool-1007-0257-f50a-58255159`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 1070, 'eta': '11h7m', 'grad': 0.1556, 'ce': 3.16}.
- f5: `mose-ffn-f5-r1-p1-1x8-pool-1007-0257-2198-70783152`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 1070, 'eta': '11h8m', 'grad': 0.2247, 'ce': 3.162}.
- a1: `mose-attn-a1-r1-p1-1x8-pool-1007-0128-5292-1-78ab4`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 2780, 'eta': '9h53m', 'grad': 0.0453, 'ce': 2.802}.
- a2: `mose-attn-a2-r1-p1-1x8-pool-1007-0128-b35f-2-1cf16`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 2790, 'eta': '9h50m', 'grad': 0.0421, 'ce': 2.856}.
- c0s2: `mose-r512-c0s2-r1-p1-1x8-pool-1006-2106-ea9c-f2516`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 7970, 'eta': '5h22m', 'grad': 0.0202, 'ce': 2.628}.
- c6s2: `mose-r512-c6s2-r1-p1-1x8-pool-1006-2107-3d78-52e71`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 7730, 'eta': '5h44m', 'grad': 0.0177, 'ce': 2.696}.
- c0s3: `mose-r512-c0s3-r1-p1-1x8-pool-1006-2107-2f04-80461`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 7920, 'eta': '5h26m', 'grad': 0.0196, 'ce': 2.688}.
- c6s3: `mose-r512-c6s3-r1-p1-1x8-pool-1006-2108-31a7-fc86e`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 7850, 'eta': '5h32m', 'grad': 0.0219, 'ce': 2.685}.
- c2s2: `mose-r512-c2s2-r1-p1-1x8-pool-1006-2338-833a-1f47a`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 4950, 'eta': '7h59m', 'grad': 0.019, 'ce': 2.716}.
- c2s3: `mose-r512-c2s3-r1-p1-1x8-pool-1006-2338-d226-b5503`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 4790, 'eta': '8h21m', 'grad': 0.0238, 'ce': 2.697}.

## 2026-10-07 04:26 HKT — monitoring

- p0: `mose-ug-p0-r1-p1-1x8-pool-1006-1123-7104-77485485`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_112904-olmo3-1b-ug-partial-p0-s0-p512-linear-down-rms-fp8-30b'].
- p4: `mose-recover-p4-r2-p1-1x8-pool-1006-1207-0d3-44098`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_120940-olmo3-1b-ug-partial-p4-s160-p368-linear-down-rms-fp8-30b'].
- p3: `mose-recover-p3-r3-p1-1x8-pool-1006-1228-c38-15484`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_123139-olmo3-1b-ug-partial-p3-s320-p224-linear-down-rms-fp8-30b'].
- p2qk: `mose-ug-p2qk-r1-p1-4x8-arch-1006-1347-8485-7-afb90`; Succeeded; {'active': 0, 'succeeded': 4, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_135053-olmo3-1b-ug-partial-p2qk-s480-p80-linear-qk-silu-down-rms-fp8-30b'].
- p1: `mose-recover-p1-r3-p1-1x8-pool-1006-1358-5ee-ff2d5`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_140134-olmo3-1b-ug-partial-p1-s560-p0-linear-down-rms-fp8-30b'].
- p0qk: `mose-ug-p0qk-r1-p1-1x8-pool-1006-1347-459e-3-74da9`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_140141-olmo3-1b-ug-partial-p0qk-s0-p512-linear-qk-silu-down-rms-fp8-30b'].
- p2: `mose-recover-p2-r3-p1-2x8-arch-1006-1409-7a8-a7469`; Succeeded; {'active': 0, 'succeeded': 2, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_141212-olmo3-1b-ug-partial-p2-s480-p80-linear-down-rms-fp8-30b'].
- p3qk: `mose-ug-p3qk-r1-p1-1x8-pool-1006-2111-2dcc-7-ec6b7`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 8030, 'eta': '5h37m', 'grad': 0.0305, 'ce': 2.674}.
- p4qk: `mose-ug-p4qk-r1-p1-1x8-pool-1006-2111-0416-8-5686b`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 8000, 'eta': '5h39m', 'grad': 0.0206, 'ce': 2.592}.
- f1: `mose-recover-f1-r2-p1-1x8-pool-1006-1207-08d-af98d`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_121016-olmo3-1b-ffn-budget-f1-ug384-down768-unshared-linear-rms-fp8-30b'].
- f2: `mose-recover-f2-r2-p1-1x8-pool-1006-1213-0e1-bdefa`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_121541-olmo3-1b-ffn-budget-f2-ug256-down1024-unshared-linear-rms-fp8-30b'].
- f3: `mose-recover-f3-r2-p1-1x8-pool-1006-1213-ccc-5d862`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_121614-olmo3-1b-ffn-budget-f3-ug480-down672-shared-linear-rms-fp8-30b'].
- f3qk: `mose-ffn-f3qk-r1-p1-4x8-arch-1006-1923-76a7-5350e`; Succeeded; {'active': 0, 'succeeded': 4, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_192653-olmo3-1b-ffn-budget-f3qk-ug480-down672-shared-qk-silu-ug-linear-down-rms-fp8-30b'].
- f2qk: `mose-ffn-f2qk-r1-p1-1x8-pool-1006-1922-88d3-ed4c5`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 9850, 'eta': '3h45m', 'grad': 0.0208, 'ce': 2.554}.
- f1qk: `mose-ffn-f1qk-r2-p1-1x8-arch-1006-1937-bfde-3c52a`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 10200, 'eta': '3h32m', 'grad': 0.0642, 'ce': 2.555}.
- f4qk: `mose-ffn-f4qk-r1-p1-2x8-arch-1006-2300-6d41-46be5`; Running; {'active': 2, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 12400, 'eta': '50m', 'grad': 0.0138, 'ce': 2.581}.
- f5qk: `mose-ffn-f5qk-r1-p1-2x8-arch-1006-2300-f278-d8613`; Running; {'active': 2, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 12610, 'eta': '43m', 'grad': 0.0128, 'ce': 2.553}.
- f4: `mose-ffn-f4-r1-p1-1x8-pool-1007-0257-f50a-58255159`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 1680, 'eta': '10h33m', 'grad': 0.0843, 'ce': 2.99}.
- f5: `mose-ffn-f5-r1-p1-1x8-pool-1007-0257-2198-70783152`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 1670, 'eta': '10h35m', 'grad': 0.1043, 'ce': 2.998}.
- a1: `mose-attn-a1-r1-p1-1x8-pool-1007-0128-5292-1-78ab4`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 3370, 'eta': '9h22m', 'grad': 0.0415, 'ce': 2.834}.
- a2: `mose-attn-a2-r1-p1-1x8-pool-1007-0128-b35f-2-1cf16`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 3390, 'eta': '9h19m', 'grad': 0.0554, 'ce': 2.777}.
- c0s2: `mose-r512-c0s2-r1-p1-1x8-pool-1006-2106-ea9c-f2516`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 8540, 'eta': '4h54m', 'grad': 0.0403, 'ce': 2.616}.
- c6s2: `mose-r512-c6s2-r1-p1-1x8-pool-1006-2107-3d78-52e71`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 8320, 'eta': '5h13m', 'grad': 0.0227, 'ce': 2.625}.
- c0s3: `mose-r512-c0s3-r1-p1-1x8-pool-1006-2107-2f04-80461`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 8490, 'eta': '4h57m', 'grad': 0.0236, 'ce': 2.658}.
- c6s3: `mose-r512-c6s3-r1-p1-1x8-pool-1006-2108-31a7-fc86e`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 8440, 'eta': '5h2m', 'grad': 0.0161, 'ce': 2.563}.
- c2s2: `mose-r512-c2s2-r1-p1-1x8-pool-1006-2338-833a-1f47a`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 5520, 'eta': '7h31m', 'grad': 0.0309, 'ce': 2.687}.
- c2s3: `mose-r512-c2s3-r1-p1-1x8-pool-1006-2338-d226-b5503`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 5370, 'eta': '7h50m', 'grad': 0.021, 'ce': 2.73}.

## 2026-10-07 04:56 HKT — monitoring

- p0: `mose-ug-p0-r1-p1-1x8-pool-1006-1123-7104-77485485`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_112904-olmo3-1b-ug-partial-p0-s0-p512-linear-down-rms-fp8-30b'].
- p4: `mose-recover-p4-r2-p1-1x8-pool-1006-1207-0d3-44098`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_120940-olmo3-1b-ug-partial-p4-s160-p368-linear-down-rms-fp8-30b'].
- p3: `mose-recover-p3-r3-p1-1x8-pool-1006-1228-c38-15484`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_123139-olmo3-1b-ug-partial-p3-s320-p224-linear-down-rms-fp8-30b'].
- p2qk: `mose-ug-p2qk-r1-p1-4x8-arch-1006-1347-8485-7-afb90`; Succeeded; {'active': 0, 'succeeded': 4, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_135053-olmo3-1b-ug-partial-p2qk-s480-p80-linear-qk-silu-down-rms-fp8-30b'].
- p1: `mose-recover-p1-r3-p1-1x8-pool-1006-1358-5ee-ff2d5`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_140134-olmo3-1b-ug-partial-p1-s560-p0-linear-down-rms-fp8-30b'].
- p0qk: `mose-ug-p0qk-r1-p1-1x8-pool-1006-1347-459e-3-74da9`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_140141-olmo3-1b-ug-partial-p0qk-s0-p512-linear-qk-silu-down-rms-fp8-30b'].
- p2: `mose-recover-p2-r3-p1-2x8-arch-1006-1409-7a8-a7469`; Succeeded; {'active': 0, 'succeeded': 2, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_141212-olmo3-1b-ug-partial-p2-s480-p80-linear-down-rms-fp8-30b'].
- p3qk: `mose-ug-p3qk-r1-p1-1x8-pool-1006-2111-2dcc-7-ec6b7`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 8590, 'eta': '5h7m', 'grad': 0.0439, 'ce': 2.635}.
- p4qk: `mose-ug-p4qk-r1-p1-1x8-pool-1006-2111-0416-8-5686b`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 8560, 'eta': '5h9m', 'grad': 0.0243, 'ce': 2.657}.
- f1: `mose-recover-f1-r2-p1-1x8-pool-1006-1207-08d-af98d`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_121016-olmo3-1b-ffn-budget-f1-ug384-down768-unshared-linear-rms-fp8-30b'].
- f2: `mose-recover-f2-r2-p1-1x8-pool-1006-1213-0e1-bdefa`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_121541-olmo3-1b-ffn-budget-f2-ug256-down1024-unshared-linear-rms-fp8-30b'].
- f3: `mose-recover-f3-r2-p1-1x8-pool-1006-1213-ccc-5d862`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_121614-olmo3-1b-ffn-budget-f3-ug480-down672-shared-linear-rms-fp8-30b'].
- f3qk: `mose-ffn-f3qk-r1-p1-4x8-arch-1006-1923-76a7-5350e`; Succeeded; {'active': 0, 'succeeded': 4, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_192653-olmo3-1b-ffn-budget-f3qk-ug480-down672-shared-qk-silu-ug-linear-down-rms-fp8-30b'].
- f2qk: `mose-ffn-f2qk-r1-p1-1x8-pool-1006-1922-88d3-ed4c5`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 10450, 'eta': '3h14m', 'grad': 0.0189, 'ce': 2.631}.
- f1qk: `mose-ffn-f1qk-r2-p1-1x8-arch-1006-1937-bfde-3c52a`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 10790, 'eta': '3h1m', 'grad': 0.0177, 'ce': 2.643}.
- f4qk: `mose-ffn-f4qk-r1-p1-2x8-arch-1006-2300-6d41-46be5`; Running; {'active': 2, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 13510, 'eta': '21m', 'grad': 0.0217, 'ce': 2.532}.
- f5qk: `mose-ffn-f5qk-r1-p1-2x8-arch-1006-2300-f278-d8613`; Running; {'active': 2, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 13810, 'eta': '13m', 'grad': 0.0109, 'ce': 2.488}.
- f4: `mose-ffn-f4-r1-p1-1x8-pool-1007-0257-f50a-58255159`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 2280, 'eta': '10h2m', 'grad': 0.0545, 'ce': 2.873}.
- f5: `mose-ffn-f5-r1-p1-1x8-pool-1007-0257-2198-70783152`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 2280, 'eta': '10h3m', 'grad': 0.0373, 'ce': 2.872}.
- a1: `mose-attn-a1-r1-p1-1x8-pool-1007-0128-5292-1-78ab4`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 3960, 'eta': '8h51m', 'grad': 0.0915, 'ce': 2.77}.
- a2: `mose-attn-a2-r1-p1-1x8-pool-1007-0128-b35f-2-1cf16`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 3970, 'eta': '8h49m', 'grad': 0.0345, 'ce': 2.781}.
- c0s2: `mose-r512-c0s2-r1-p1-1x8-pool-1006-2106-ea9c-f2516`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 9130, 'eta': '4h24m', 'grad': 0.0213, 'ce': 2.586}.
- c6s2: `mose-r512-c6s2-r1-p1-1x8-pool-1006-2107-3d78-52e71`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 8900, 'eta': '4h43m', 'grad': 0.022, 'ce': 2.692}.
- c0s3: `mose-r512-c0s3-r1-p1-1x8-pool-1006-2107-2f04-80461`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 9080, 'eta': '4h27m', 'grad': 0.0194, 'ce': 2.624}.
- c6s3: `mose-r512-c6s3-r1-p1-1x8-pool-1006-2108-31a7-fc86e`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 9000, 'eta': '4h34m', 'grad': 0.02, 'ce': 2.62}.
- c2s2: `mose-r512-c2s2-r1-p1-1x8-pool-1006-2338-833a-1f47a`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 6110, 'eta': '7h', 'grad': 0.0332, 'ce': 2.615}.
- c2s3: `mose-r512-c2s3-r1-p1-1x8-pool-1006-2338-d226-b5503`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 5950, 'eta': '7h19m', 'grad': 0.018, 'ce': 2.571}.

## 2026-10-07 05:26 HKT — monitoring

- p0: `mose-ug-p0-r1-p1-1x8-pool-1006-1123-7104-77485485`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_112904-olmo3-1b-ug-partial-p0-s0-p512-linear-down-rms-fp8-30b'].
- p4: `mose-recover-p4-r2-p1-1x8-pool-1006-1207-0d3-44098`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_120940-olmo3-1b-ug-partial-p4-s160-p368-linear-down-rms-fp8-30b'].
- p3: `mose-recover-p3-r3-p1-1x8-pool-1006-1228-c38-15484`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_123139-olmo3-1b-ug-partial-p3-s320-p224-linear-down-rms-fp8-30b'].
- p2qk: `mose-ug-p2qk-r1-p1-4x8-arch-1006-1347-8485-7-afb90`; Succeeded; {'active': 0, 'succeeded': 4, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_135053-olmo3-1b-ug-partial-p2qk-s480-p80-linear-qk-silu-down-rms-fp8-30b'].
- p1: `mose-recover-p1-r3-p1-1x8-pool-1006-1358-5ee-ff2d5`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_140134-olmo3-1b-ug-partial-p1-s560-p0-linear-down-rms-fp8-30b'].
- p0qk: `mose-ug-p0qk-r1-p1-1x8-pool-1006-1347-459e-3-74da9`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_140141-olmo3-1b-ug-partial-p0qk-s0-p512-linear-qk-silu-down-rms-fp8-30b'].
- p2: `mose-recover-p2-r3-p1-2x8-arch-1006-1409-7a8-a7469`; Succeeded; {'active': 0, 'succeeded': 2, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_141212-olmo3-1b-ug-partial-p2-s480-p80-linear-down-rms-fp8-30b'].
- p3qk: `mose-ug-p3qk-r1-p1-1x8-pool-1006-2111-2dcc-7-ec6b7`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 9150, 'eta': '4h36m', 'grad': 0.0235, 'ce': 2.639}.
- p4qk: `mose-ug-p4qk-r1-p1-1x8-pool-1006-2111-0416-8-5686b`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 9130, 'eta': '4h38m', 'grad': 0.0155, 'ce': 2.589}.
- f1: `mose-recover-f1-r2-p1-1x8-pool-1006-1207-08d-af98d`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_121016-olmo3-1b-ffn-budget-f1-ug384-down768-unshared-linear-rms-fp8-30b'].
- f2: `mose-recover-f2-r2-p1-1x8-pool-1006-1213-0e1-bdefa`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_121541-olmo3-1b-ffn-budget-f2-ug256-down1024-unshared-linear-rms-fp8-30b'].
- f3: `mose-recover-f3-r2-p1-1x8-pool-1006-1213-ccc-5d862`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_121614-olmo3-1b-ffn-budget-f3-ug480-down672-shared-linear-rms-fp8-30b'].
- f3qk: `mose-ffn-f3qk-r1-p1-4x8-arch-1006-1923-76a7-5350e`; Succeeded; {'active': 0, 'succeeded': 4, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_192653-olmo3-1b-ffn-budget-f3qk-ug480-down672-shared-qk-silu-ug-linear-down-rms-fp8-30b'].
- f2qk: `mose-ffn-f2qk-r1-p1-1x8-pool-1006-1922-88d3-ed4c5`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 11030, 'eta': '2h46m', 'grad': 0.0163, 'ce': 2.681}.
- f1qk: `mose-ffn-f1qk-r2-p1-1x8-arch-1006-1937-bfde-3c52a`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 11370, 'eta': '2h31m', 'grad': 0.0254, 'ce': 2.612}.
- f4qk: `mose-ffn-f4qk-r1-p1-2x8-arch-1006-2300-6d41-46be5`; Succeeded; {'active': 0, 'succeeded': 2, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 14300, 'eta': '1m', 'grad': 0.0069, 'ce': 2.508}.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_230506-olmo3-1b-ffn-budget-f4qk-ug400-down816-shared-qk-silu-ug-linear-down-rms-fp8-30b', '/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_230506-olmo3-1b-ffn-budget-f4qk-ug400-down816-shared-qk-silu-ug-linear-down-rms-fp8-30b/logs'].
- f5qk: `mose-ffn-f5qk-r1-p1-2x8-arch-1006-2300-f278-d8613`; Succeeded; {'active': 0, 'succeeded': 2, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 14300, 'eta': '1m', 'grad': 0.0068, 'ce': 2.502}.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_230543-olmo3-1b-ffn-budget-f5qk-ug560-down528-shared-qk-silu-ug-linear-down-rms-fp8-30b', '/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_230543-olmo3-1b-ffn-budget-f5qk-ug560-down528-shared-qk-silu-ug-linear-down-rms-fp8-30b/logs'].
- f4: `mose-ffn-f4-r1-p1-1x8-pool-1007-0257-f50a-58255159`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 2880, 'eta': '9h31m', 'grad': 0.0485, 'ce': 2.894}.
- f5: `mose-ffn-f5-r1-p1-1x8-pool-1007-0257-2198-70783152`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 2880, 'eta': '9h32m', 'grad': 0.0621, 'ce': 2.893}.
- a1: `mose-attn-a1-r1-p1-1x8-pool-1007-0128-5292-1-78ab4`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 4520, 'eta': '8h25m', 'grad': 0.0456, 'ce': 2.674}.
- a2: `mose-attn-a2-r1-p1-1x8-pool-1007-0128-b35f-2-1cf16`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 4540, 'eta': '8h22m', 'grad': 0.0641, 'ce': 2.74}.
- c0s2: `mose-r512-c0s2-r1-p1-1x8-pool-1006-2106-ea9c-f2516`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 9730, 'eta': '3h53m', 'grad': 0.0228, 'ce': 2.64}.
- c6s2: `mose-r512-c6s2-r1-p1-1x8-pool-1006-2107-3d78-52e71`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 9470, 'eta': '4h13m', 'grad': 0.0206, 'ce': 2.61}.
- c0s3: `mose-r512-c0s3-r1-p1-1x8-pool-1006-2107-2f04-80461`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 9670, 'eta': '3h57m', 'grad': 0.0199, 'ce': 2.647}.
- c6s3: `mose-r512-c6s3-r1-p1-1x8-pool-1006-2108-31a7-fc86e`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 9590, 'eta': '4h3m', 'grad': 0.0234, 'ce': 2.623}.
- c2s2: `mose-r512-c2s2-r1-p1-1x8-pool-1006-2338-833a-1f47a`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 6710, 'eta': '6h29m', 'grad': 0.0177, 'ce': 2.605}.
- c2s3: `mose-r512-c2s3-r1-p1-1x8-pool-1006-2338-d226-b5503`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 6500, 'eta': '6h51m', 'grad': 0.021, 'ce': 2.625}.

## 2026-10-07 05:56 HKT — monitoring

- p0: `mose-ug-p0-r1-p1-1x8-pool-1006-1123-7104-77485485`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_112904-olmo3-1b-ug-partial-p0-s0-p512-linear-down-rms-fp8-30b'].
- p4: `mose-recover-p4-r2-p1-1x8-pool-1006-1207-0d3-44098`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_120940-olmo3-1b-ug-partial-p4-s160-p368-linear-down-rms-fp8-30b'].
- p3: `mose-recover-p3-r3-p1-1x8-pool-1006-1228-c38-15484`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_123139-olmo3-1b-ug-partial-p3-s320-p224-linear-down-rms-fp8-30b'].
- p2qk: `mose-ug-p2qk-r1-p1-4x8-arch-1006-1347-8485-7-afb90`; Succeeded; {'active': 0, 'succeeded': 4, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_135053-olmo3-1b-ug-partial-p2qk-s480-p80-linear-qk-silu-down-rms-fp8-30b'].
- p1: `mose-recover-p1-r3-p1-1x8-pool-1006-1358-5ee-ff2d5`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_140134-olmo3-1b-ug-partial-p1-s560-p0-linear-down-rms-fp8-30b'].
- p0qk: `mose-ug-p0qk-r1-p1-1x8-pool-1006-1347-459e-3-74da9`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_140141-olmo3-1b-ug-partial-p0qk-s0-p512-linear-qk-silu-down-rms-fp8-30b'].
- p2: `mose-recover-p2-r3-p1-2x8-arch-1006-1409-7a8-a7469`; Succeeded; {'active': 0, 'succeeded': 2, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_141212-olmo3-1b-ug-partial-p2-s480-p80-linear-down-rms-fp8-30b'].
- p3qk: `mose-ug-p3qk-r1-p1-1x8-pool-1006-2111-2dcc-7-ec6b7`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 9720, 'eta': '4h6m', 'grad': 0.08, 'ce': 2.545}.
- p4qk: `mose-ug-p4qk-r1-p1-1x8-pool-1006-2111-0416-8-5686b`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 9690, 'eta': '4h8m', 'grad': 0.0228, 'ce': 2.615}.
- f1: `mose-recover-f1-r2-p1-1x8-pool-1006-1207-08d-af98d`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_121016-olmo3-1b-ffn-budget-f1-ug384-down768-unshared-linear-rms-fp8-30b'].
- f2: `mose-recover-f2-r2-p1-1x8-pool-1006-1213-0e1-bdefa`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_121541-olmo3-1b-ffn-budget-f2-ug256-down1024-unshared-linear-rms-fp8-30b'].
- f3: `mose-recover-f3-r2-p1-1x8-pool-1006-1213-ccc-5d862`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_121614-olmo3-1b-ffn-budget-f3-ug480-down672-shared-linear-rms-fp8-30b'].
- f3qk: `mose-ffn-f3qk-r1-p1-4x8-arch-1006-1923-76a7-5350e`; Succeeded; {'active': 0, 'succeeded': 4, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_192653-olmo3-1b-ffn-budget-f3qk-ug480-down672-shared-qk-silu-ug-linear-down-rms-fp8-30b'].
- f2qk: `mose-ffn-f2qk-r1-p1-1x8-pool-1006-1922-88d3-ed4c5`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 11630, 'eta': '2h15m', 'grad': 0.018, 'ce': 2.633}.
- f1qk: `mose-ffn-f1qk-r2-p1-1x8-arch-1006-1937-bfde-3c52a`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 11960, 'eta': '2h1m', 'grad': 0.0309, 'ce': 2.636}.
- f4qk: `mose-ffn-f4qk-r1-p1-2x8-arch-1006-2300-6d41-46be5`; Succeeded; {'active': 0, 'succeeded': 2, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_230506-olmo3-1b-ffn-budget-f4qk-ug400-down816-shared-qk-silu-ug-linear-down-rms-fp8-30b'].
- f5qk: `mose-ffn-f5qk-r1-p1-2x8-arch-1006-2300-f278-d8613`; Succeeded; {'active': 0, 'succeeded': 2, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_230543-olmo3-1b-ffn-budget-f5qk-ug560-down528-shared-qk-silu-ug-linear-down-rms-fp8-30b'].
- f4: `mose-ffn-f4-r1-p1-1x8-pool-1007-0257-f50a-58255159`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 3490, 'eta': '9h', 'grad': 0.0382, 'ce': 2.833}.
- f5: `mose-ffn-f5-r1-p1-1x8-pool-1007-0257-2198-70783152`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 3490, 'eta': '9h', 'grad': 0.0471, 'ce': 2.831}.
- a1: `mose-attn-a1-r1-p1-1x8-pool-1007-0128-5292-1-78ab4`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 5110, 'eta': '7h54m', 'grad': 0.0518, 'ce': 2.704}.
- a2: `mose-attn-a2-r1-p1-1x8-pool-1007-0128-b35f-2-1cf16`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 5130, 'eta': '7h51m', 'grad': 0.0188, 'ce': 2.769}.
- c0s2: `mose-r512-c0s2-r1-p1-1x8-pool-1006-2106-ea9c-f2516`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 10320, 'eta': '3h23m', 'grad': 0.0301, 'ce': 2.62}.
- c6s2: `mose-r512-c6s2-r1-p1-1x8-pool-1006-2107-3d78-52e71`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 10030, 'eta': '3h44m', 'grad': 0.0147, 'ce': 2.591}.
- c0s3: `mose-r512-c0s3-r1-p1-1x8-pool-1006-2107-2f04-80461`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 10260, 'eta': '3h27m', 'grad': 0.0219, 'ce': 2.638}.
- c6s3: `mose-r512-c6s3-r1-p1-1x8-pool-1006-2108-31a7-fc86e`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 10180, 'eta': '3h33m', 'grad': 0.0171, 'ce': 2.624}.
- c2s2: `mose-r512-c2s2-r1-p1-1x8-pool-1006-2338-833a-1f47a`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 7300, 'eta': '5h59m', 'grad': 0.0273, 'ce': 2.65}.
- c2s3: `mose-r512-c2s3-r1-p1-1x8-pool-1006-2338-d226-b5503`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 7070, 'eta': '6h21m', 'grad': 0.0182, 'ce': 2.656}.

## 2026-10-07 06:26 HKT — monitoring

- p0: `mose-ug-p0-r1-p1-1x8-pool-1006-1123-7104-77485485`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_112904-olmo3-1b-ug-partial-p0-s0-p512-linear-down-rms-fp8-30b'].
- p4: `mose-recover-p4-r2-p1-1x8-pool-1006-1207-0d3-44098`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_120940-olmo3-1b-ug-partial-p4-s160-p368-linear-down-rms-fp8-30b'].
- p3: `mose-recover-p3-r3-p1-1x8-pool-1006-1228-c38-15484`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_123139-olmo3-1b-ug-partial-p3-s320-p224-linear-down-rms-fp8-30b'].
- p2qk: `mose-ug-p2qk-r1-p1-4x8-arch-1006-1347-8485-7-afb90`; Succeeded; {'active': 0, 'succeeded': 4, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_135053-olmo3-1b-ug-partial-p2qk-s480-p80-linear-qk-silu-down-rms-fp8-30b'].
- p1: `mose-recover-p1-r3-p1-1x8-pool-1006-1358-5ee-ff2d5`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_140134-olmo3-1b-ug-partial-p1-s560-p0-linear-down-rms-fp8-30b'].
- p0qk: `mose-ug-p0qk-r1-p1-1x8-pool-1006-1347-459e-3-74da9`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_140141-olmo3-1b-ug-partial-p0qk-s0-p512-linear-qk-silu-down-rms-fp8-30b'].
- p2: `mose-recover-p2-r3-p1-2x8-arch-1006-1409-7a8-a7469`; Succeeded; {'active': 0, 'succeeded': 2, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_141212-olmo3-1b-ug-partial-p2-s480-p80-linear-down-rms-fp8-30b'].
- p3qk: `mose-ug-p3qk-r1-p1-1x8-pool-1006-2111-2dcc-7-ec6b7`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 10280, 'eta': '3h36m', 'grad': 0.0624, 'ce': 2.63}.
- p4qk: `mose-ug-p4qk-r1-p1-1x8-pool-1006-2111-0416-8-5686b`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 10250, 'eta': '3h38m', 'grad': 0.0146, 'ce': 2.55}.
- f1: `mose-recover-f1-r2-p1-1x8-pool-1006-1207-08d-af98d`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_121016-olmo3-1b-ffn-budget-f1-ug384-down768-unshared-linear-rms-fp8-30b'].
- f2: `mose-recover-f2-r2-p1-1x8-pool-1006-1213-0e1-bdefa`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_121541-olmo3-1b-ffn-budget-f2-ug256-down1024-unshared-linear-rms-fp8-30b'].
- f3: `mose-recover-f3-r2-p1-1x8-pool-1006-1213-ccc-5d862`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_121614-olmo3-1b-ffn-budget-f3-ug480-down672-shared-linear-rms-fp8-30b'].
- f3qk: `mose-ffn-f3qk-r1-p1-4x8-arch-1006-1923-76a7-5350e`; Succeeded; {'active': 0, 'succeeded': 4, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_192653-olmo3-1b-ffn-budget-f3qk-ug480-down672-shared-qk-silu-ug-linear-down-rms-fp8-30b'].
- f2qk: `mose-ffn-f2qk-r1-p1-1x8-pool-1006-1922-88d3-ed4c5`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 12230, 'eta': '1h45m', 'grad': 0.024, 'ce': 2.542}.
- f1qk: `mose-ffn-f1qk-r2-p1-1x8-arch-1006-1937-bfde-3c52a`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 12510, 'eta': '1h33m', 'grad': 0.0361, 'ce': 2.559}.
- f4qk: `mose-ffn-f4qk-r1-p1-2x8-arch-1006-2300-6d41-46be5`; Succeeded; {'active': 0, 'succeeded': 2, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_230506-olmo3-1b-ffn-budget-f4qk-ug400-down816-shared-qk-silu-ug-linear-down-rms-fp8-30b'].
- f5qk: `mose-ffn-f5qk-r1-p1-2x8-arch-1006-2300-f278-d8613`; Succeeded; {'active': 0, 'succeeded': 2, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_230543-olmo3-1b-ffn-budget-f5qk-ug560-down528-shared-qk-silu-ug-linear-down-rms-fp8-30b'].
- f4: `mose-ffn-f4-r1-p1-1x8-pool-1007-0257-f50a-58255159`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 4070, 'eta': '8h34m', 'grad': 0.0426, 'ce': 2.788}.
- f5: `mose-ffn-f5-r1-p1-1x8-pool-1007-0257-2198-70783152`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 4070, 'eta': '8h34m', 'grad': 0.0235, 'ce': 2.786}.
- a1: `mose-attn-a1-r1-p1-1x8-pool-1007-0128-5292-1-78ab4`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 5700, 'eta': '7h23m', 'grad': 0.0699, 'ce': 2.623}.
- a2: `mose-attn-a2-r1-p1-1x8-pool-1007-0128-b35f-2-1cf16`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 5720, 'eta': '7h20m', 'grad': 0.0357, 'ce': 2.696}.
- c0s2: `mose-r512-c0s2-r1-p1-1x8-pool-1006-2106-ea9c-f2516`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 10910, 'eta': '2h53m', 'grad': 0.018, 'ce': 2.608}.
- c6s2: `mose-r512-c6s2-r1-p1-1x8-pool-1006-2107-3d78-52e71`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 10610, 'eta': '3h14m', 'grad': 0.0206, 'ce': 2.573}.
- c0s3: `mose-r512-c0s3-r1-p1-1x8-pool-1006-2107-2f04-80461`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 10860, 'eta': '2h56m', 'grad': 0.0172, 'ce': 2.606}.
- c6s3: `mose-r512-c6s3-r1-p1-1x8-pool-1006-2108-31a7-fc86e`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 10770, 'eta': '3h2m', 'grad': 0.0167, 'ce': 2.538}.
- c2s2: `mose-r512-c2s2-r1-p1-1x8-pool-1006-2338-833a-1f47a`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 7890, 'eta': '5h28m', 'grad': 0.0173, 'ce': 2.653}.
- c2s3: `mose-r512-c2s3-r1-p1-1x8-pool-1006-2338-d226-b5503`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 7650, 'eta': '5h50m', 'grad': 0.0174, 'ce': 2.624}.

## 2026-10-07 06:56 HKT — monitoring

- p0: `mose-ug-p0-r1-p1-1x8-pool-1006-1123-7104-77485485`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_112904-olmo3-1b-ug-partial-p0-s0-p512-linear-down-rms-fp8-30b'].
- p4: `mose-recover-p4-r2-p1-1x8-pool-1006-1207-0d3-44098`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_120940-olmo3-1b-ug-partial-p4-s160-p368-linear-down-rms-fp8-30b'].
- p3: `mose-recover-p3-r3-p1-1x8-pool-1006-1228-c38-15484`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_123139-olmo3-1b-ug-partial-p3-s320-p224-linear-down-rms-fp8-30b'].
- p2qk: `mose-ug-p2qk-r1-p1-4x8-arch-1006-1347-8485-7-afb90`; Succeeded; {'active': 0, 'succeeded': 4, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_135053-olmo3-1b-ug-partial-p2qk-s480-p80-linear-qk-silu-down-rms-fp8-30b'].
- p1: `mose-recover-p1-r3-p1-1x8-pool-1006-1358-5ee-ff2d5`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_140134-olmo3-1b-ug-partial-p1-s560-p0-linear-down-rms-fp8-30b'].
- p0qk: `mose-ug-p0qk-r1-p1-1x8-pool-1006-1347-459e-3-74da9`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_140141-olmo3-1b-ug-partial-p0qk-s0-p512-linear-qk-silu-down-rms-fp8-30b'].
- p2: `mose-recover-p2-r3-p1-2x8-arch-1006-1409-7a8-a7469`; Succeeded; {'active': 0, 'succeeded': 2, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_141212-olmo3-1b-ug-partial-p2-s480-p80-linear-down-rms-fp8-30b'].
- p3qk: `mose-ug-p3qk-r1-p1-1x8-pool-1006-2111-2dcc-7-ec6b7`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 10850, 'eta': '3h5m', 'grad': 0.0341, 'ce': 2.613}.
- p4qk: `mose-ug-p4qk-r1-p1-1x8-pool-1006-2111-0416-8-5686b`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 10810, 'eta': '3h8m', 'grad': 0.0187, 'ce': 2.667}.
- f1: `mose-recover-f1-r2-p1-1x8-pool-1006-1207-08d-af98d`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_121016-olmo3-1b-ffn-budget-f1-ug384-down768-unshared-linear-rms-fp8-30b'].
- f2: `mose-recover-f2-r2-p1-1x8-pool-1006-1213-0e1-bdefa`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_121541-olmo3-1b-ffn-budget-f2-ug256-down1024-unshared-linear-rms-fp8-30b'].
- f3: `mose-recover-f3-r2-p1-1x8-pool-1006-1213-ccc-5d862`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_121614-olmo3-1b-ffn-budget-f3-ug480-down672-shared-linear-rms-fp8-30b'].
- f3qk: `mose-ffn-f3qk-r1-p1-4x8-arch-1006-1923-76a7-5350e`; Succeeded; {'active': 0, 'succeeded': 4, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_192653-olmo3-1b-ffn-budget-f3qk-ug480-down672-shared-qk-silu-ug-linear-down-rms-fp8-30b'].
- f2qk: `mose-ffn-f2qk-r1-p1-1x8-pool-1006-1922-88d3-ed4c5`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 12810, 'eta': '1h16m', 'grad': 0.0135, 'ce': 2.535}.
- f1qk: `mose-ffn-f1qk-r2-p1-1x8-arch-1006-1937-bfde-3c52a`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 13090, 'eta': '1h3m', 'grad': 0.0266, 'ce': 2.535}.
- f4qk: `mose-ffn-f4qk-r1-p1-2x8-arch-1006-2300-6d41-46be5`; Succeeded; {'active': 0, 'succeeded': 2, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_230506-olmo3-1b-ffn-budget-f4qk-ug400-down816-shared-qk-silu-ug-linear-down-rms-fp8-30b'].
- f5qk: `mose-ffn-f5qk-r1-p1-2x8-arch-1006-2300-f278-d8613`; Succeeded; {'active': 0, 'succeeded': 2, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_230543-olmo3-1b-ffn-budget-f5qk-ug560-down528-shared-qk-silu-ug-linear-down-rms-fp8-30b'].
- f4: `mose-ffn-f4-r1-p1-1x8-pool-1007-0257-f50a-58255159`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 4670, 'eta': '8h3m', 'grad': 0.028, 'ce': 2.712}.
- f5: `mose-ffn-f5-r1-p1-1x8-pool-1007-0257-2198-70783152`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 4670, 'eta': '8h3m', 'grad': 0.026, 'ce': 2.711}.
- a1: `mose-attn-a1-r1-p1-1x8-pool-1007-0128-5292-1-78ab4`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 6290, 'eta': '6h52m', 'grad': 0.0399, 'ce': 2.666}.
- a2: `mose-attn-a2-r1-p1-1x8-pool-1007-0128-b35f-2-1cf16`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 6310, 'eta': '6h50m', 'grad': 0.0154, 'ce': 2.726}.
- c0s2: `mose-r512-c0s2-r1-p1-1x8-pool-1006-2106-ea9c-f2516`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 11490, 'eta': '2h23m', 'grad': 0.0219, 'ce': 2.587}.
- c6s2: `mose-r512-c6s2-r1-p1-1x8-pool-1006-2107-3d78-52e71`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 11190, 'eta': '2h43m', 'grad': 0.0189, 'ce': 2.648}.
- c0s3: `mose-r512-c0s3-r1-p1-1x8-pool-1006-2107-2f04-80461`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 11450, 'eta': '2h26m', 'grad': 0.0168, 'ce': 2.597}.
- c6s3: `mose-r512-c6s3-r1-p1-1x8-pool-1006-2108-31a7-fc86e`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 11360, 'eta': '2h32m', 'grad': 0.0282, 'ce': 2.621}.
- c2s2: `mose-r512-c2s2-r1-p1-1x8-pool-1006-2338-833a-1f47a`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 8480, 'eta': '4h58m', 'grad': 0.0196, 'ce': 2.628}.
- c2s3: `mose-r512-c2s3-r1-p1-1x8-pool-1006-2338-d226-b5503`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 8230, 'eta': '5h19m', 'grad': 0.0212, 'ce': 2.615}.

## 2026-10-07 07:26 HKT — monitoring

- p0: `mose-ug-p0-r1-p1-1x8-pool-1006-1123-7104-77485485`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_112904-olmo3-1b-ug-partial-p0-s0-p512-linear-down-rms-fp8-30b'].
- p4: `mose-recover-p4-r2-p1-1x8-pool-1006-1207-0d3-44098`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_120940-olmo3-1b-ug-partial-p4-s160-p368-linear-down-rms-fp8-30b'].
- p3: `mose-recover-p3-r3-p1-1x8-pool-1006-1228-c38-15484`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_123139-olmo3-1b-ug-partial-p3-s320-p224-linear-down-rms-fp8-30b'].
- p2qk: `mose-ug-p2qk-r1-p1-4x8-arch-1006-1347-8485-7-afb90`; Succeeded; {'active': 0, 'succeeded': 4, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_135053-olmo3-1b-ug-partial-p2qk-s480-p80-linear-qk-silu-down-rms-fp8-30b'].
- p1: `mose-recover-p1-r3-p1-1x8-pool-1006-1358-5ee-ff2d5`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_140134-olmo3-1b-ug-partial-p1-s560-p0-linear-down-rms-fp8-30b'].
- p0qk: `mose-ug-p0qk-r1-p1-1x8-pool-1006-1347-459e-3-74da9`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_140141-olmo3-1b-ug-partial-p0qk-s0-p512-linear-qk-silu-down-rms-fp8-30b'].
- p2: `mose-recover-p2-r3-p1-2x8-arch-1006-1409-7a8-a7469`; Succeeded; {'active': 0, 'succeeded': 2, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_141212-olmo3-1b-ug-partial-p2-s480-p80-linear-down-rms-fp8-30b'].
- p3qk: `mose-ug-p3qk-r1-p1-1x8-pool-1006-2111-2dcc-7-ec6b7`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 11410, 'eta': '2h35m', 'grad': 0.0427, 'ce': 2.649}.
- p4qk: `mose-ug-p4qk-r1-p1-1x8-pool-1006-2111-0416-8-5686b`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 11380, 'eta': '2h37m', 'grad': 0.0213, 'ce': 2.635}.
- f1: `mose-recover-f1-r2-p1-1x8-pool-1006-1207-08d-af98d`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_121016-olmo3-1b-ffn-budget-f1-ug384-down768-unshared-linear-rms-fp8-30b'].
- f2: `mose-recover-f2-r2-p1-1x8-pool-1006-1213-0e1-bdefa`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_121541-olmo3-1b-ffn-budget-f2-ug256-down1024-unshared-linear-rms-fp8-30b'].
- f3: `mose-recover-f3-r2-p1-1x8-pool-1006-1213-ccc-5d862`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_121614-olmo3-1b-ffn-budget-f3-ug480-down672-shared-linear-rms-fp8-30b'].
- f3qk: `mose-ffn-f3qk-r1-p1-4x8-arch-1006-1923-76a7-5350e`; Succeeded; {'active': 0, 'succeeded': 4, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_192653-olmo3-1b-ffn-budget-f3qk-ug480-down672-shared-qk-silu-ug-linear-down-rms-fp8-30b'].
- f2qk: `mose-ffn-f2qk-r1-p1-1x8-pool-1006-1922-88d3-ed4c5`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 13410, 'eta': '46m', 'grad': 0.0123, 'ce': 2.507}.
- f1qk: `mose-ffn-f1qk-r2-p1-1x8-arch-1006-1937-bfde-3c52a`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 13680, 'eta': '33m', 'grad': 0.0143, 'ce': 2.433}.
- f4qk: `mose-ffn-f4qk-r1-p1-2x8-arch-1006-2300-6d41-46be5`; Succeeded; {'active': 0, 'succeeded': 2, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_230506-olmo3-1b-ffn-budget-f4qk-ug400-down816-shared-qk-silu-ug-linear-down-rms-fp8-30b'].
- f5qk: `mose-ffn-f5qk-r1-p1-2x8-arch-1006-2300-f278-d8613`; Succeeded; {'active': 0, 'succeeded': 2, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_230543-olmo3-1b-ffn-budget-f5qk-ug560-down528-shared-qk-silu-ug-linear-down-rms-fp8-30b'].
- f4: `mose-ffn-f4-r1-p1-1x8-pool-1007-0257-f50a-58255159`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 5280, 'eta': '7h32m', 'grad': 0.0363, 'ce': 2.713}.
- f5: `mose-ffn-f5-r1-p1-1x8-pool-1007-0257-2198-70783152`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 5280, 'eta': '7h32m', 'grad': 0.0204, 'ce': 2.71}.
- a1: `mose-attn-a1-r1-p1-1x8-pool-1007-0128-5292-1-78ab4`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 6870, 'eta': '6h22m', 'grad': 0.0259, 'ce': 2.661}.
- a2: `mose-attn-a2-r1-p1-1x8-pool-1007-0128-b35f-2-1cf16`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 6900, 'eta': '6h19m', 'grad': 0.0235, 'ce': 2.674}.
- c0s2: `mose-r512-c0s2-r1-p1-1x8-pool-1006-2106-ea9c-f2516`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 12080, 'eta': '1h54m', 'grad': 0.0178, 'ce': 2.594}.
- c6s2: `mose-r512-c6s2-r1-p1-1x8-pool-1006-2107-3d78-52e71`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 11770, 'eta': '2h13m', 'grad': 0.0174, 'ce': 2.521}.
- c0s3: `mose-r512-c0s3-r1-p1-1x8-pool-1006-2107-2f04-80461`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 12020, 'eta': '1h57m', 'grad': 0.0184, 'ce': 2.489}.
- c6s3: `mose-r512-c6s3-r1-p1-1x8-pool-1006-2108-31a7-fc86e`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 11950, 'eta': '2h1m', 'grad': 0.0275, 'ce': 2.554}.
- c2s2: `mose-r512-c2s2-r1-p1-1x8-pool-1006-2338-833a-1f47a`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 9050, 'eta': '4h29m', 'grad': 0.0175, 'ce': 2.613}.
- c2s3: `mose-r512-c2s3-r1-p1-1x8-pool-1006-2338-d226-b5503`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 8800, 'eta': '4h49m', 'grad': 0.0228, 'ce': 2.65}.

## 2026-10-07 07:56 HKT — monitoring

- p0: `mose-ug-p0-r1-p1-1x8-pool-1006-1123-7104-77485485`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_112904-olmo3-1b-ug-partial-p0-s0-p512-linear-down-rms-fp8-30b'].
- p4: `mose-recover-p4-r2-p1-1x8-pool-1006-1207-0d3-44098`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_120940-olmo3-1b-ug-partial-p4-s160-p368-linear-down-rms-fp8-30b'].
- p3: `mose-recover-p3-r3-p1-1x8-pool-1006-1228-c38-15484`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_123139-olmo3-1b-ug-partial-p3-s320-p224-linear-down-rms-fp8-30b'].
- p2qk: `mose-ug-p2qk-r1-p1-4x8-arch-1006-1347-8485-7-afb90`; Succeeded; {'active': 0, 'succeeded': 4, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_135053-olmo3-1b-ug-partial-p2qk-s480-p80-linear-qk-silu-down-rms-fp8-30b'].
- p1: `mose-recover-p1-r3-p1-1x8-pool-1006-1358-5ee-ff2d5`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_140134-olmo3-1b-ug-partial-p1-s560-p0-linear-down-rms-fp8-30b'].
- p0qk: `mose-ug-p0qk-r1-p1-1x8-pool-1006-1347-459e-3-74da9`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_140141-olmo3-1b-ug-partial-p0qk-s0-p512-linear-qk-silu-down-rms-fp8-30b'].
- p2: `mose-recover-p2-r3-p1-2x8-arch-1006-1409-7a8-a7469`; Succeeded; {'active': 0, 'succeeded': 2, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_141212-olmo3-1b-ug-partial-p2-s480-p80-linear-down-rms-fp8-30b'].
- p3qk: `mose-ug-p3qk-r1-p1-1x8-pool-1006-2111-2dcc-7-ec6b7`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 11980, 'eta': '2h5m', 'grad': 0.0125, 'ce': 2.619}.
- p4qk: `mose-ug-p4qk-r1-p1-1x8-pool-1006-2111-0416-8-5686b`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 11940, 'eta': '2h7m', 'grad': 0.0185, 'ce': 2.607}.
- f1: `mose-recover-f1-r2-p1-1x8-pool-1006-1207-08d-af98d`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_121016-olmo3-1b-ffn-budget-f1-ug384-down768-unshared-linear-rms-fp8-30b'].
- f2: `mose-recover-f2-r2-p1-1x8-pool-1006-1213-0e1-bdefa`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_121541-olmo3-1b-ffn-budget-f2-ug256-down1024-unshared-linear-rms-fp8-30b'].
- f3: `mose-recover-f3-r2-p1-1x8-pool-1006-1213-ccc-5d862`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_121614-olmo3-1b-ffn-budget-f3-ug480-down672-shared-linear-rms-fp8-30b'].
- f3qk: `mose-ffn-f3qk-r1-p1-4x8-arch-1006-1923-76a7-5350e`; Succeeded; {'active': 0, 'succeeded': 4, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_192653-olmo3-1b-ffn-budget-f3qk-ug480-down672-shared-qk-silu-ug-linear-down-rms-fp8-30b'].
- f2qk: `mose-ffn-f2qk-r1-p1-1x8-pool-1006-1922-88d3-ed4c5`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 13990, 'eta': '16m', 'grad': 0.0131, 'ce': 2.543}.
- f1qk: `mose-ffn-f1qk-r2-p1-1x8-arch-1006-1937-bfde-3c52a`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 14260, 'eta': '3m', 'grad': 0.0273, 'ce': 2.445}.
- f4qk: `mose-ffn-f4qk-r1-p1-2x8-arch-1006-2300-6d41-46be5`; Succeeded; {'active': 0, 'succeeded': 2, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_230506-olmo3-1b-ffn-budget-f4qk-ug400-down816-shared-qk-silu-ug-linear-down-rms-fp8-30b'].
- f5qk: `mose-ffn-f5qk-r1-p1-2x8-arch-1006-2300-f278-d8613`; Succeeded; {'active': 0, 'succeeded': 2, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_230543-olmo3-1b-ffn-budget-f5qk-ug560-down528-shared-qk-silu-ug-linear-down-rms-fp8-30b'].
- f4: `mose-ffn-f4-r1-p1-1x8-pool-1007-0257-f50a-58255159`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 5880, 'eta': '7h1m', 'grad': 0.0292, 'ce': 2.668}.
- f5: `mose-ffn-f5-r1-p1-1x8-pool-1007-0257-2198-70783152`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 5880, 'eta': '7h2m', 'grad': 0.0355, 'ce': 2.666}.
- a1: `mose-attn-a1-r1-p1-1x8-pool-1007-0128-5292-1-78ab4`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 7460, 'eta': '5h52m', 'grad': 0.0179, 'ce': 2.639}.
- a2: `mose-attn-a2-r1-p1-1x8-pool-1007-0128-b35f-2-1cf16`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 7490, 'eta': '5h49m', 'grad': 0.0837, 'ce': 2.646}.
- c0s2: `mose-r512-c0s2-r1-p1-1x8-pool-1006-2106-ea9c-f2516`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 12650, 'eta': '1h25m', 'grad': 0.0181, 'ce': 2.571}.
- c6s2: `mose-r512-c6s2-r1-p1-1x8-pool-1006-2107-3d78-52e71`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 12350, 'eta': '1h42m', 'grad': 0.0122, 'ce': 2.515}.
- c0s3: `mose-r512-c0s3-r1-p1-1x8-pool-1006-2107-2f04-80461`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 12590, 'eta': '1h28m', 'grad': 0.0161, 'ce': 2.593}.
- c6s3: `mose-r512-c6s3-r1-p1-1x8-pool-1006-2108-31a7-fc86e`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
- c2s2: `mose-r512-c2s2-r1-p1-1x8-pool-1006-2338-833a-1f47a`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 9650, 'eta': '3h59m', 'grad': 0.0136, 'ce': 2.601}.
- c2s3: `mose-r512-c2s3-r1-p1-1x8-pool-1006-2338-d226-b5503`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 9380, 'eta': '4h19m', 'grad': 0.0257, 'ce': 2.556}.

## 2026-10-07 08:26 HKT — monitoring

- p0: `mose-ug-p0-r1-p1-1x8-pool-1006-1123-7104-77485485`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_112904-olmo3-1b-ug-partial-p0-s0-p512-linear-down-rms-fp8-30b'].
- p4: `mose-recover-p4-r2-p1-1x8-pool-1006-1207-0d3-44098`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_120940-olmo3-1b-ug-partial-p4-s160-p368-linear-down-rms-fp8-30b'].
- p3: `mose-recover-p3-r3-p1-1x8-pool-1006-1228-c38-15484`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_123139-olmo3-1b-ug-partial-p3-s320-p224-linear-down-rms-fp8-30b'].
- p2qk: `mose-ug-p2qk-r1-p1-4x8-arch-1006-1347-8485-7-afb90`; Succeeded; {'active': 0, 'succeeded': 4, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_135053-olmo3-1b-ug-partial-p2qk-s480-p80-linear-qk-silu-down-rms-fp8-30b'].
- p1: `mose-recover-p1-r3-p1-1x8-pool-1006-1358-5ee-ff2d5`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_140134-olmo3-1b-ug-partial-p1-s560-p0-linear-down-rms-fp8-30b'].
- p0qk: `mose-ug-p0qk-r1-p1-1x8-pool-1006-1347-459e-3-74da9`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_140141-olmo3-1b-ug-partial-p0qk-s0-p512-linear-qk-silu-down-rms-fp8-30b'].
- p2: `mose-recover-p2-r3-p1-2x8-arch-1006-1409-7a8-a7469`; Succeeded; {'active': 0, 'succeeded': 2, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_141212-olmo3-1b-ug-partial-p2-s480-p80-linear-down-rms-fp8-30b'].
- p3qk: `mose-ug-p3qk-r1-p1-1x8-pool-1006-2111-2dcc-7-ec6b7`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
- p4qk: `mose-ug-p4qk-r1-p1-1x8-pool-1006-2111-0416-8-5686b`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 12490, 'eta': '1h38m', 'grad': 0.0152, 'ce': 2.529}.
- f1: `mose-recover-f1-r2-p1-1x8-pool-1006-1207-08d-af98d`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_121016-olmo3-1b-ffn-budget-f1-ug384-down768-unshared-linear-rms-fp8-30b'].
- f2: `mose-recover-f2-r2-p1-1x8-pool-1006-1213-0e1-bdefa`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_121541-olmo3-1b-ffn-budget-f2-ug256-down1024-unshared-linear-rms-fp8-30b'].
- f3: `mose-recover-f3-r2-p1-1x8-pool-1006-1213-ccc-5d862`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_121614-olmo3-1b-ffn-budget-f3-ug480-down672-shared-linear-rms-fp8-30b'].
- f3qk: `mose-ffn-f3qk-r1-p1-4x8-arch-1006-1923-76a7-5350e`; Succeeded; {'active': 0, 'succeeded': 4, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_192653-olmo3-1b-ffn-budget-f3qk-ug480-down672-shared-qk-silu-ug-linear-down-rms-fp8-30b'].
- f2qk: `mose-ffn-f2qk-r1-p1-1x8-pool-1006-1922-88d3-ed4c5`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 14300, 'eta': '1m', 'grad': 0.0072, 'ce': 2.518}.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_200818-olmo3-1b-ffn-budget-f2qk-ug256-down1024-unshared-qk-silu-ug-linear-down-rms-fp8-30b', '/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_200818-olmo3-1b-ffn-budget-f2qk-ug256-down1024-unshared-qk-silu-ug-linear-down-rms-fp8-30b/logs'].
- f1qk: `mose-ffn-f1qk-r2-p1-1x8-arch-1006-1937-bfde-3c52a`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 14300, 'eta': '1m', 'grad': 0.0074, 'ce': 2.505}.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_193938-olmo3-1b-ffn-budget-f1qk-ug384-down768-unshared-qk-silu-ug-linear-down-rms-fp8-30b', '/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_193938-olmo3-1b-ffn-budget-f1qk-ug384-down768-unshared-qk-silu-ug-linear-down-rms-fp8-30b/logs'].
- f4qk: `mose-ffn-f4qk-r1-p1-2x8-arch-1006-2300-6d41-46be5`; Succeeded; {'active': 0, 'succeeded': 2, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_230506-olmo3-1b-ffn-budget-f4qk-ug400-down816-shared-qk-silu-ug-linear-down-rms-fp8-30b'].
- f5qk: `mose-ffn-f5qk-r1-p1-2x8-arch-1006-2300-f278-d8613`; Succeeded; {'active': 0, 'succeeded': 2, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_230543-olmo3-1b-ffn-budget-f5qk-ug560-down528-shared-qk-silu-ug-linear-down-rms-fp8-30b'].
- f4: `mose-ffn-f4-r1-p1-1x8-pool-1007-0257-f50a-58255159`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 6490, 'eta': '6h31m', 'grad': 0.0183, 'ce': 2.619}.
- f5: `mose-ffn-f5-r1-p1-1x8-pool-1007-0257-2198-70783152`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 6480, 'eta': '6h32m', 'grad': 0.0188, 'ce': 2.606}.
- a1: `mose-attn-a1-r1-p1-1x8-pool-1007-0128-5292-1-78ab4`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 8020, 'eta': '5h24m', 'grad': 0.0266, 'ce': 2.755}.
- a2: `mose-attn-a2-r1-p1-1x8-pool-1007-0128-b35f-2-1cf16`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 8060, 'eta': '5h21m', 'grad': 0.0789, 'ce': 2.666}.
- c0s2: `mose-r512-c0s2-r1-p1-1x8-pool-1006-2106-ea9c-f2516`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 13240, 'eta': '55m', 'grad': 0.0172, 'ce': 2.487}.
- c6s2: `mose-r512-c6s2-r1-p1-1x8-pool-1006-2107-3d78-52e71`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 12910, 'eta': '1h13m', 'grad': 0.0141, 'ce': 2.504}.
- c0s3: `mose-r512-c0s3-r1-p1-1x8-pool-1006-2107-2f04-80461`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 13180, 'eta': '58m', 'grad': 0.0111, 'ce': 2.492}.
- c6s3: `mose-r512-c6s3-r1-p1-1x8-pool-1006-2108-31a7-fc86e`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 13080, 'eta': '1h4m', 'grad': 0.0131, 'ce': 2.52}.
- c2s2: `mose-r512-c2s2-r1-p1-1x8-pool-1006-2338-833a-1f47a`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 10240, 'eta': '3h28m', 'grad': 0.0208, 'ce': 2.61}.
- c2s3: `mose-r512-c2s3-r1-p1-1x8-pool-1006-2338-d226-b5503`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 9960, 'eta': '3h48m', 'grad': 0.0223, 'ce': 2.617}.

## 2026-10-07 08:56 HKT — monitoring

- p0: `mose-ug-p0-r1-p1-1x8-pool-1006-1123-7104-77485485`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_112904-olmo3-1b-ug-partial-p0-s0-p512-linear-down-rms-fp8-30b'].
- p4: `mose-recover-p4-r2-p1-1x8-pool-1006-1207-0d3-44098`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_120940-olmo3-1b-ug-partial-p4-s160-p368-linear-down-rms-fp8-30b'].
- p3: `mose-recover-p3-r3-p1-1x8-pool-1006-1228-c38-15484`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_123139-olmo3-1b-ug-partial-p3-s320-p224-linear-down-rms-fp8-30b'].
- p2qk: `mose-ug-p2qk-r1-p1-4x8-arch-1006-1347-8485-7-afb90`; Succeeded; {'active': 0, 'succeeded': 4, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_135053-olmo3-1b-ug-partial-p2qk-s480-p80-linear-qk-silu-down-rms-fp8-30b'].
- p1: `mose-recover-p1-r3-p1-1x8-pool-1006-1358-5ee-ff2d5`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_140134-olmo3-1b-ug-partial-p1-s560-p0-linear-down-rms-fp8-30b'].
- p0qk: `mose-ug-p0qk-r1-p1-1x8-pool-1006-1347-459e-3-74da9`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_140141-olmo3-1b-ug-partial-p0qk-s0-p512-linear-qk-silu-down-rms-fp8-30b'].
- p2: `mose-recover-p2-r3-p1-2x8-arch-1006-1409-7a8-a7469`; Succeeded; {'active': 0, 'succeeded': 2, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_141212-olmo3-1b-ug-partial-p2-s480-p80-linear-down-rms-fp8-30b'].
- p3qk: `mose-ug-p3qk-r1-p1-1x8-pool-1006-2111-2dcc-7-ec6b7`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 13060, 'eta': '1h7m', 'grad': 0.0607, 'ce': 2.498}.
- p4qk: `mose-ug-p4qk-r1-p1-1x8-pool-1006-2111-0416-8-5686b`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 13020, 'eta': '1h10m', 'grad': 0.0189, 'ce': 2.561}.
- f1: `mose-recover-f1-r2-p1-1x8-pool-1006-1207-08d-af98d`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_121016-olmo3-1b-ffn-budget-f1-ug384-down768-unshared-linear-rms-fp8-30b'].
- f2: `mose-recover-f2-r2-p1-1x8-pool-1006-1213-0e1-bdefa`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_121541-olmo3-1b-ffn-budget-f2-ug256-down1024-unshared-linear-rms-fp8-30b'].
- f3: `mose-recover-f3-r2-p1-1x8-pool-1006-1213-ccc-5d862`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_121614-olmo3-1b-ffn-budget-f3-ug480-down672-shared-linear-rms-fp8-30b'].
- f3qk: `mose-ffn-f3qk-r1-p1-4x8-arch-1006-1923-76a7-5350e`; Succeeded; {'active': 0, 'succeeded': 4, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_192653-olmo3-1b-ffn-budget-f3qk-ug480-down672-shared-qk-silu-ug-linear-down-rms-fp8-30b'].
- f2qk: `mose-ffn-f2qk-r1-p1-1x8-pool-1006-1922-88d3-ed4c5`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_200818-olmo3-1b-ffn-budget-f2qk-ug256-down1024-unshared-qk-silu-ug-linear-down-rms-fp8-30b'].
- f1qk: `mose-ffn-f1qk-r2-p1-1x8-arch-1006-1937-bfde-3c52a`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_193938-olmo3-1b-ffn-budget-f1qk-ug384-down768-unshared-qk-silu-ug-linear-down-rms-fp8-30b'].
- f4qk: `mose-ffn-f4qk-r1-p1-2x8-arch-1006-2300-6d41-46be5`; Succeeded; {'active': 0, 'succeeded': 2, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_230506-olmo3-1b-ffn-budget-f4qk-ug400-down816-shared-qk-silu-ug-linear-down-rms-fp8-30b'].
- f5qk: `mose-ffn-f5qk-r1-p1-2x8-arch-1006-2300-f278-d8613`; Succeeded; {'active': 0, 'succeeded': 2, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_230543-olmo3-1b-ffn-budget-f5qk-ug560-down528-shared-qk-silu-ug-linear-down-rms-fp8-30b'].
- f4: `mose-ffn-f4-r1-p1-1x8-pool-1007-0257-f50a-58255159`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 7070, 'eta': '6h3m', 'grad': 0.0195, 'ce': 2.667}.
- f5: `mose-ffn-f5-r1-p1-1x8-pool-1007-0257-2198-70783152`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 7060, 'eta': '6h4m', 'grad': 0.0213, 'ce': 2.667}.
- a1: `mose-attn-a1-r1-p1-1x8-pool-1007-0128-5292-1-78ab4`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 8610, 'eta': '4h54m', 'grad': 0.0169, 'ce': 2.674}.
- a2: `mose-attn-a2-r1-p1-1x8-pool-1007-0128-b35f-2-1cf16`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 8650, 'eta': '4h50m', 'grad': 0.0791, 'ce': 2.632}.
- c0s2: `mose-r512-c0s2-r1-p1-1x8-pool-1006-2106-ea9c-f2516`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 13840, 'eta': '24m', 'grad': 0.0105, 'ce': 2.532}.
- c6s2: `mose-r512-c6s2-r1-p1-1x8-pool-1006-2107-3d78-52e71`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 13490, 'eta': '43m', 'grad': 0.0138, 'ce': 2.523}.
- c0s3: `mose-r512-c0s3-r1-p1-1x8-pool-1006-2107-2f04-80461`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 13770, 'eta': '28m', 'grad': 0.0112, 'ce': 2.436}.
- c6s3: `mose-r512-c6s3-r1-p1-1x8-pool-1006-2108-31a7-fc86e`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 13660, 'eta': '34m', 'grad': 0.0117, 'ce': 2.432}.
- c2s2: `mose-r512-c2s2-r1-p1-1x8-pool-1006-2338-833a-1f47a`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 10830, 'eta': '2h58m', 'grad': 0.0199, 'ce': 2.552}.
- c2s3: `mose-r512-c2s3-r1-p1-1x8-pool-1006-2338-d226-b5503`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 10500, 'eta': '3h20m', 'grad': 0.0186, 'ce': 2.602}.

## 2026-10-07 09:26 HKT — monitoring

- p0: `mose-ug-p0-r1-p1-1x8-pool-1006-1123-7104-77485485`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_112904-olmo3-1b-ug-partial-p0-s0-p512-linear-down-rms-fp8-30b'].
- p4: `mose-recover-p4-r2-p1-1x8-pool-1006-1207-0d3-44098`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_120940-olmo3-1b-ug-partial-p4-s160-p368-linear-down-rms-fp8-30b'].
- p3: `mose-recover-p3-r3-p1-1x8-pool-1006-1228-c38-15484`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_123139-olmo3-1b-ug-partial-p3-s320-p224-linear-down-rms-fp8-30b'].
- p2qk: `mose-ug-p2qk-r1-p1-4x8-arch-1006-1347-8485-7-afb90`; Succeeded; {'active': 0, 'succeeded': 4, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_135053-olmo3-1b-ug-partial-p2qk-s480-p80-linear-qk-silu-down-rms-fp8-30b'].
- p1: `mose-recover-p1-r3-p1-1x8-pool-1006-1358-5ee-ff2d5`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_140134-olmo3-1b-ug-partial-p1-s560-p0-linear-down-rms-fp8-30b'].
- p0qk: `mose-ug-p0qk-r1-p1-1x8-pool-1006-1347-459e-3-74da9`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_140141-olmo3-1b-ug-partial-p0qk-s0-p512-linear-qk-silu-down-rms-fp8-30b'].
- p2: `mose-recover-p2-r3-p1-2x8-arch-1006-1409-7a8-a7469`; Succeeded; {'active': 0, 'succeeded': 2, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_141212-olmo3-1b-ug-partial-p2-s480-p80-linear-down-rms-fp8-30b'].
- p3qk: `mose-ug-p3qk-r1-p1-1x8-pool-1006-2111-2dcc-7-ec6b7`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 13620, 'eta': '37m', 'grad': 0.0259, 'ce': 2.485}.
- p4qk: `mose-ug-p4qk-r1-p1-1x8-pool-1006-2111-0416-8-5686b`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 13580, 'eta': '39m', 'grad': 0.0116, 'ce': 2.472}.
- f1: `mose-recover-f1-r2-p1-1x8-pool-1006-1207-08d-af98d`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_121016-olmo3-1b-ffn-budget-f1-ug384-down768-unshared-linear-rms-fp8-30b'].
- f2: `mose-recover-f2-r2-p1-1x8-pool-1006-1213-0e1-bdefa`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_121541-olmo3-1b-ffn-budget-f2-ug256-down1024-unshared-linear-rms-fp8-30b'].
- f3: `mose-recover-f3-r2-p1-1x8-pool-1006-1213-ccc-5d862`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_121614-olmo3-1b-ffn-budget-f3-ug480-down672-shared-linear-rms-fp8-30b'].
- f3qk: `mose-ffn-f3qk-r1-p1-4x8-arch-1006-1923-76a7-5350e`; Succeeded; {'active': 0, 'succeeded': 4, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_192653-olmo3-1b-ffn-budget-f3qk-ug480-down672-shared-qk-silu-ug-linear-down-rms-fp8-30b'].
- f2qk: `mose-ffn-f2qk-r1-p1-1x8-pool-1006-1922-88d3-ed4c5`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_200818-olmo3-1b-ffn-budget-f2qk-ug256-down1024-unshared-qk-silu-ug-linear-down-rms-fp8-30b'].
- f1qk: `mose-ffn-f1qk-r2-p1-1x8-arch-1006-1937-bfde-3c52a`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_193938-olmo3-1b-ffn-budget-f1qk-ug384-down768-unshared-qk-silu-ug-linear-down-rms-fp8-30b'].
- f4qk: `mose-ffn-f4qk-r1-p1-2x8-arch-1006-2300-6d41-46be5`; Succeeded; {'active': 0, 'succeeded': 2, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_230506-olmo3-1b-ffn-budget-f4qk-ug400-down816-shared-qk-silu-ug-linear-down-rms-fp8-30b'].
- f5qk: `mose-ffn-f5qk-r1-p1-2x8-arch-1006-2300-f278-d8613`; Succeeded; {'active': 0, 'succeeded': 2, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_230543-olmo3-1b-ffn-budget-f5qk-ug560-down528-shared-qk-silu-ug-linear-down-rms-fp8-30b'].
- f4: `mose-ffn-f4-r1-p1-1x8-pool-1007-0257-f50a-58255159`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 7670, 'eta': '5h32m', 'grad': 0.0169, 'ce': 2.685}.
- f5: `mose-ffn-f5-r1-p1-1x8-pool-1007-0257-2198-70783152`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 7670, 'eta': '5h33m', 'grad': 0.0221, 'ce': 2.682}.
- a1: `mose-attn-a1-r1-p1-1x8-pool-1007-0128-5292-1-78ab4`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 9200, 'eta': '4h23m', 'grad': 0.0208, 'ce': 2.591}.
- a2: `mose-attn-a2-r1-p1-1x8-pool-1007-0128-b35f-2-1cf16`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 9240, 'eta': '4h20m', 'grad': 0.0573, 'ce': 2.641}.
- c0s2: `mose-r512-c0s2-r1-p1-1x8-pool-1006-2106-ea9c-f2516`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 14300, 'eta': '1m', 'grad': 0.008, 'ce': 2.502}.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_210958-olmo3-1b-r512matched-c0-seed12537-data34521-fp8-30b', '/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_210958-olmo3-1b-r512matched-c0-seed12537-data34521-fp8-30b/logs'].
- c6s2: `mose-r512-c6s2-r1-p1-1x8-pool-1006-2107-3d78-52e71`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 14040, 'eta': '14m', 'grad': 0.0097, 'ce': 2.501}.
- c0s3: `mose-r512-c0s3-r1-p1-1x8-pool-1006-2107-2f04-80461`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 14300, 'eta': '1m', 'grad': 0.0071, 'ce': 2.502}.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_211054-olmo3-1b-r512matched-c0-seed12538-data34521-fp8-30b', '/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_211054-olmo3-1b-r512matched-c0-seed12538-data34521-fp8-30b/logs'].
- c6s3: `mose-r512-c6s3-r1-p1-1x8-pool-1006-2108-31a7-fc86e`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 14250, 'eta': '3m', 'grad': 0.0079, 'ce': 2.489}.
- c2s2: `mose-r512-c2s2-r1-p1-1x8-pool-1006-2338-833a-1f47a`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 11420, 'eta': '2h28m', 'grad': 0.0251, 'ce': 2.614}.
- c2s3: `mose-r512-c2s3-r1-p1-1x8-pool-1006-2338-d226-b5503`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 11080, 'eta': '2h50m', 'grad': 0.0182, 'ce': 2.62}.

## 2026-10-07 09:56 HKT — monitoring

- p0: `mose-ug-p0-r1-p1-1x8-pool-1006-1123-7104-77485485`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_112904-olmo3-1b-ug-partial-p0-s0-p512-linear-down-rms-fp8-30b'].
- p4: `mose-recover-p4-r2-p1-1x8-pool-1006-1207-0d3-44098`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_120940-olmo3-1b-ug-partial-p4-s160-p368-linear-down-rms-fp8-30b'].
- p3: `mose-recover-p3-r3-p1-1x8-pool-1006-1228-c38-15484`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_123139-olmo3-1b-ug-partial-p3-s320-p224-linear-down-rms-fp8-30b'].
- p2qk: `mose-ug-p2qk-r1-p1-4x8-arch-1006-1347-8485-7-afb90`; Succeeded; {'active': 0, 'succeeded': 4, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_135053-olmo3-1b-ug-partial-p2qk-s480-p80-linear-qk-silu-down-rms-fp8-30b'].
- p1: `mose-recover-p1-r3-p1-1x8-pool-1006-1358-5ee-ff2d5`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_140134-olmo3-1b-ug-partial-p1-s560-p0-linear-down-rms-fp8-30b'].
- p0qk: `mose-ug-p0qk-r1-p1-1x8-pool-1006-1347-459e-3-74da9`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_140141-olmo3-1b-ug-partial-p0qk-s0-p512-linear-qk-silu-down-rms-fp8-30b'].
- p2: `mose-recover-p2-r3-p1-2x8-arch-1006-1409-7a8-a7469`; Succeeded; {'active': 0, 'succeeded': 2, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_141212-olmo3-1b-ug-partial-p2-s480-p80-linear-down-rms-fp8-30b'].
- p3qk: `mose-ug-p3qk-r1-p1-1x8-pool-1006-2111-2dcc-7-ec6b7`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 14180, 'eta': '7m', 'grad': 0.0112, 'ce': 2.518}.
- p4qk: `mose-ug-p4qk-r1-p1-1x8-pool-1006-2111-0416-8-5686b`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 14140, 'eta': '9m', 'grad': 0.0098, 'ce': 2.494}.
- f1: `mose-recover-f1-r2-p1-1x8-pool-1006-1207-08d-af98d`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_121016-olmo3-1b-ffn-budget-f1-ug384-down768-unshared-linear-rms-fp8-30b'].
- f2: `mose-recover-f2-r2-p1-1x8-pool-1006-1213-0e1-bdefa`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_121541-olmo3-1b-ffn-budget-f2-ug256-down1024-unshared-linear-rms-fp8-30b'].
- f3: `mose-recover-f3-r2-p1-1x8-pool-1006-1213-ccc-5d862`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_121614-olmo3-1b-ffn-budget-f3-ug480-down672-shared-linear-rms-fp8-30b'].
- f3qk: `mose-ffn-f3qk-r1-p1-4x8-arch-1006-1923-76a7-5350e`; Succeeded; {'active': 0, 'succeeded': 4, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_192653-olmo3-1b-ffn-budget-f3qk-ug480-down672-shared-qk-silu-ug-linear-down-rms-fp8-30b'].
- f2qk: `mose-ffn-f2qk-r1-p1-1x8-pool-1006-1922-88d3-ed4c5`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_200818-olmo3-1b-ffn-budget-f2qk-ug256-down1024-unshared-qk-silu-ug-linear-down-rms-fp8-30b'].
- f1qk: `mose-ffn-f1qk-r2-p1-1x8-arch-1006-1937-bfde-3c52a`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_193938-olmo3-1b-ffn-budget-f1qk-ug384-down768-unshared-qk-silu-ug-linear-down-rms-fp8-30b'].
- f4qk: `mose-ffn-f4qk-r1-p1-2x8-arch-1006-2300-6d41-46be5`; Succeeded; {'active': 0, 'succeeded': 2, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_230506-olmo3-1b-ffn-budget-f4qk-ug400-down816-shared-qk-silu-ug-linear-down-rms-fp8-30b'].
- f5qk: `mose-ffn-f5qk-r1-p1-2x8-arch-1006-2300-f278-d8613`; Succeeded; {'active': 0, 'succeeded': 2, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_230543-olmo3-1b-ffn-budget-f5qk-ug560-down528-shared-qk-silu-ug-linear-down-rms-fp8-30b'].
- f4: `mose-ffn-f4-r1-p1-1x8-pool-1007-0257-f50a-58255159`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 8280, 'eta': '5h2m', 'grad': 0.032, 'ce': 2.663}.
- f5: `mose-ffn-f5-r1-p1-1x8-pool-1007-0257-2198-70783152`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 8270, 'eta': '5h3m', 'grad': 0.03, 'ce': 2.695}.
- a1: `mose-attn-a1-r1-p1-1x8-pool-1007-0128-5292-1-78ab4`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 9780, 'eta': '3h53m', 'grad': 0.036, 'ce': 2.579}.
- a2: `mose-attn-a2-r1-p1-1x8-pool-1007-0128-b35f-2-1cf16`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 9830, 'eta': '3h50m', 'grad': 0.0655, 'ce': 2.709}.
- c0s2: `mose-r512-c0s2-r1-p1-1x8-pool-1006-2106-ea9c-f2516`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_210958-olmo3-1b-r512matched-c0-seed12537-data34521-fp8-30b'].
- c6s2: `mose-r512-c6s2-r1-p1-1x8-pool-1006-2107-3d78-52e71`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 14300, 'eta': '1m', 'grad': 0.007, 'ce': 2.498}.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_211039-olmo3-1b-r512matched-c6-seed12537-data34521-fp8-30b', '/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_211039-olmo3-1b-r512matched-c6-seed12537-data34521-fp8-30b/logs'].
- c0s3: `mose-r512-c0s3-r1-p1-1x8-pool-1006-2107-2f04-80461`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_211054-olmo3-1b-r512matched-c0-seed12538-data34521-fp8-30b'].
- c6s3: `mose-r512-c6s3-r1-p1-1x8-pool-1006-2108-31a7-fc86e`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 14300, 'eta': '1m', 'grad': 0.0085, 'ce': 2.498}.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_211118-olmo3-1b-r512matched-c6-seed12538-data34521-fp8-30b', '/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_211118-olmo3-1b-r512matched-c6-seed12538-data34521-fp8-30b/logs'].
- c2s2: `mose-r512-c2s2-r1-p1-1x8-pool-1006-2338-833a-1f47a`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 11990, 'eta': '1h59m', 'grad': 0.0183, 'ce': 2.581}.
- c2s3: `mose-r512-c2s3-r1-p1-1x8-pool-1006-2338-d226-b5503`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 11660, 'eta': '2h19m', 'grad': 0.0186, 'ce': 2.566}.

## 2026-10-07 10:26 HKT — monitoring

- p0: `mose-ug-p0-r1-p1-1x8-pool-1006-1123-7104-77485485`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_112904-olmo3-1b-ug-partial-p0-s0-p512-linear-down-rms-fp8-30b'].
- p4: `mose-recover-p4-r2-p1-1x8-pool-1006-1207-0d3-44098`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_120940-olmo3-1b-ug-partial-p4-s160-p368-linear-down-rms-fp8-30b'].
- p3: `mose-recover-p3-r3-p1-1x8-pool-1006-1228-c38-15484`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_123139-olmo3-1b-ug-partial-p3-s320-p224-linear-down-rms-fp8-30b'].
- p2qk: `mose-ug-p2qk-r1-p1-4x8-arch-1006-1347-8485-7-afb90`; Succeeded; {'active': 0, 'succeeded': 4, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_135053-olmo3-1b-ug-partial-p2qk-s480-p80-linear-qk-silu-down-rms-fp8-30b'].
- p1: `mose-recover-p1-r3-p1-1x8-pool-1006-1358-5ee-ff2d5`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_140134-olmo3-1b-ug-partial-p1-s560-p0-linear-down-rms-fp8-30b'].
- p0qk: `mose-ug-p0qk-r1-p1-1x8-pool-1006-1347-459e-3-74da9`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_140141-olmo3-1b-ug-partial-p0qk-s0-p512-linear-qk-silu-down-rms-fp8-30b'].
- p2: `mose-recover-p2-r3-p1-2x8-arch-1006-1409-7a8-a7469`; Succeeded; {'active': 0, 'succeeded': 2, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_141212-olmo3-1b-ug-partial-p2-s480-p80-linear-down-rms-fp8-30b'].
- p3qk: `mose-ug-p3qk-r1-p1-1x8-pool-1006-2111-2dcc-7-ec6b7`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 14300, 'eta': '1m', 'grad': 0.007, 'ce': 2.505}.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_211408-olmo3-1b-ug-partial-p3qk-s320-p224-linear-qk-silu-down-rms-fp8-30b', '/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_211408-olmo3-1b-ug-partial-p3qk-s320-p224-linear-qk-silu-down-rms-fp8-30b/logs'].
- p4qk: `mose-ug-p4qk-r1-p1-1x8-pool-1006-2111-0416-8-5686b`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 14300, 'eta': '1m', 'grad': 0.0069, 'ce': 2.504}.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_211424-olmo3-1b-ug-partial-p4qk-s160-p368-linear-qk-silu-down-rms-fp8-30b', '/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_211424-olmo3-1b-ug-partial-p4qk-s160-p368-linear-qk-silu-down-rms-fp8-30b/logs'].
- f1: `mose-recover-f1-r2-p1-1x8-pool-1006-1207-08d-af98d`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_121016-olmo3-1b-ffn-budget-f1-ug384-down768-unshared-linear-rms-fp8-30b'].
- f2: `mose-recover-f2-r2-p1-1x8-pool-1006-1213-0e1-bdefa`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_121541-olmo3-1b-ffn-budget-f2-ug256-down1024-unshared-linear-rms-fp8-30b'].
- f3: `mose-recover-f3-r2-p1-1x8-pool-1006-1213-ccc-5d862`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_121614-olmo3-1b-ffn-budget-f3-ug480-down672-shared-linear-rms-fp8-30b'].
- f3qk: `mose-ffn-f3qk-r1-p1-4x8-arch-1006-1923-76a7-5350e`; Succeeded; {'active': 0, 'succeeded': 4, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_192653-olmo3-1b-ffn-budget-f3qk-ug480-down672-shared-qk-silu-ug-linear-down-rms-fp8-30b'].
- f2qk: `mose-ffn-f2qk-r1-p1-1x8-pool-1006-1922-88d3-ed4c5`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_200818-olmo3-1b-ffn-budget-f2qk-ug256-down1024-unshared-qk-silu-ug-linear-down-rms-fp8-30b'].
- f1qk: `mose-ffn-f1qk-r2-p1-1x8-arch-1006-1937-bfde-3c52a`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_193938-olmo3-1b-ffn-budget-f1qk-ug384-down768-unshared-qk-silu-ug-linear-down-rms-fp8-30b'].
- f4qk: `mose-ffn-f4qk-r1-p1-2x8-arch-1006-2300-6d41-46be5`; Succeeded; {'active': 0, 'succeeded': 2, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_230506-olmo3-1b-ffn-budget-f4qk-ug400-down816-shared-qk-silu-ug-linear-down-rms-fp8-30b'].
- f5qk: `mose-ffn-f5qk-r1-p1-2x8-arch-1006-2300-f278-d8613`; Succeeded; {'active': 0, 'succeeded': 2, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_230543-olmo3-1b-ffn-budget-f5qk-ug560-down528-shared-qk-silu-ug-linear-down-rms-fp8-30b'].
- f4: `mose-ffn-f4-r1-p1-1x8-pool-1007-0257-f50a-58255159`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 8880, 'eta': '4h32m', 'grad': 0.0156, 'ce': 2.626}.
- f5: `mose-ffn-f5-r1-p1-1x8-pool-1007-0257-2198-70783152`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 8880, 'eta': '4h32m', 'grad': 0.0215, 'ce': 2.622}.
- a1: `mose-attn-a1-r1-p1-1x8-pool-1007-0128-5292-1-78ab4`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 10370, 'eta': '3h23m', 'grad': 0.0342, 'ce': 2.679}.
- a2: `mose-attn-a2-r1-p1-1x8-pool-1007-0128-b35f-2-1cf16`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 10420, 'eta': '3h19m', 'grad': 0.029, 'ce': 2.619}.
- c0s2: `mose-r512-c0s2-r1-p1-1x8-pool-1006-2106-ea9c-f2516`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_210958-olmo3-1b-r512matched-c0-seed12537-data34521-fp8-30b'].
- c6s2: `mose-r512-c6s2-r1-p1-1x8-pool-1006-2107-3d78-52e71`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_211039-olmo3-1b-r512matched-c6-seed12537-data34521-fp8-30b'].
- c0s3: `mose-r512-c0s3-r1-p1-1x8-pool-1006-2107-2f04-80461`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_211054-olmo3-1b-r512matched-c0-seed12538-data34521-fp8-30b'].
- c6s3: `mose-r512-c6s3-r1-p1-1x8-pool-1006-2108-31a7-fc86e`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_211118-olmo3-1b-r512matched-c6-seed12538-data34521-fp8-30b'].
- c2s2: `mose-r512-c2s2-r1-p1-1x8-pool-1006-2338-833a-1f47a`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 12560, 'eta': '1h30m', 'grad': 0.0137, 'ce': 2.556}.
- c2s3: `mose-r512-c2s3-r1-p1-1x8-pool-1006-2338-d226-b5503`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 12230, 'eta': '1h49m', 'grad': 0.0174, 'ce': 2.525}.

## 2026-10-07 10:32 HKT — completed P3qk/P4qk analysis

Both source attempts verified Succeeded/active0, final14306 saved; exact W&B uploads Finished syncing. Analysis `mose-report-ug-qk-r7-p1-1x8-pool-1007-1029-e-47f42` Succeeded/active0/succeeded1; both payload SHA256 checks passed. P3qk/P4qk last1000 mean Train CE 2.481997/2.481281, both worse than C6 2.474763. Frozen round7 manifest:17 current nonseed configs, plus3 archived references; P1 excluded from strict797050880-parameter ranking. Full results: P3_P4_QK_RESULTS.md.

## 2026-10-07 10:56 HKT — monitoring

- p0: `mose-ug-p0-r1-p1-1x8-pool-1006-1123-7104-77485485`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_112904-olmo3-1b-ug-partial-p0-s0-p512-linear-down-rms-fp8-30b'].
- p4: `mose-recover-p4-r2-p1-1x8-pool-1006-1207-0d3-44098`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_120940-olmo3-1b-ug-partial-p4-s160-p368-linear-down-rms-fp8-30b'].
- p3: `mose-recover-p3-r3-p1-1x8-pool-1006-1228-c38-15484`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_123139-olmo3-1b-ug-partial-p3-s320-p224-linear-down-rms-fp8-30b'].
- p2qk: `mose-ug-p2qk-r1-p1-4x8-arch-1006-1347-8485-7-afb90`; Succeeded; {'active': 0, 'succeeded': 4, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_135053-olmo3-1b-ug-partial-p2qk-s480-p80-linear-qk-silu-down-rms-fp8-30b'].
- p1: `mose-recover-p1-r3-p1-1x8-pool-1006-1358-5ee-ff2d5`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_140134-olmo3-1b-ug-partial-p1-s560-p0-linear-down-rms-fp8-30b'].
- p0qk: `mose-ug-p0qk-r1-p1-1x8-pool-1006-1347-459e-3-74da9`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_140141-olmo3-1b-ug-partial-p0qk-s0-p512-linear-qk-silu-down-rms-fp8-30b'].
- p2: `mose-recover-p2-r3-p1-2x8-arch-1006-1409-7a8-a7469`; Succeeded; {'active': 0, 'succeeded': 2, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_141212-olmo3-1b-ug-partial-p2-s480-p80-linear-down-rms-fp8-30b'].
- p3qk: `mose-ug-p3qk-r1-p1-1x8-pool-1006-2111-2dcc-7-ec6b7`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_211408-olmo3-1b-ug-partial-p3qk-s320-p224-linear-qk-silu-down-rms-fp8-30b'].
- p4qk: `mose-ug-p4qk-r1-p1-1x8-pool-1006-2111-0416-8-5686b`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_211424-olmo3-1b-ug-partial-p4qk-s160-p368-linear-qk-silu-down-rms-fp8-30b'].
- f1: `mose-recover-f1-r2-p1-1x8-pool-1006-1207-08d-af98d`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_121016-olmo3-1b-ffn-budget-f1-ug384-down768-unshared-linear-rms-fp8-30b'].
- f2: `mose-recover-f2-r2-p1-1x8-pool-1006-1213-0e1-bdefa`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_121541-olmo3-1b-ffn-budget-f2-ug256-down1024-unshared-linear-rms-fp8-30b'].
- f3: `mose-recover-f3-r2-p1-1x8-pool-1006-1213-ccc-5d862`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_121614-olmo3-1b-ffn-budget-f3-ug480-down672-shared-linear-rms-fp8-30b'].
- f3qk: `mose-ffn-f3qk-r1-p1-4x8-arch-1006-1923-76a7-5350e`; Succeeded; {'active': 0, 'succeeded': 4, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_192653-olmo3-1b-ffn-budget-f3qk-ug480-down672-shared-qk-silu-ug-linear-down-rms-fp8-30b'].
- f2qk: `mose-ffn-f2qk-r1-p1-1x8-pool-1006-1922-88d3-ed4c5`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_200818-olmo3-1b-ffn-budget-f2qk-ug256-down1024-unshared-qk-silu-ug-linear-down-rms-fp8-30b'].
- f1qk: `mose-ffn-f1qk-r2-p1-1x8-arch-1006-1937-bfde-3c52a`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_193938-olmo3-1b-ffn-budget-f1qk-ug384-down768-unshared-qk-silu-ug-linear-down-rms-fp8-30b'].
- f4qk: `mose-ffn-f4qk-r1-p1-2x8-arch-1006-2300-6d41-46be5`; Succeeded; {'active': 0, 'succeeded': 2, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_230506-olmo3-1b-ffn-budget-f4qk-ug400-down816-shared-qk-silu-ug-linear-down-rms-fp8-30b'].
- f5qk: `mose-ffn-f5qk-r1-p1-2x8-arch-1006-2300-f278-d8613`; Succeeded; {'active': 0, 'succeeded': 2, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_230543-olmo3-1b-ffn-budget-f5qk-ug560-down528-shared-qk-silu-ug-linear-down-rms-fp8-30b'].
- f4: `mose-ffn-f4-r1-p1-1x8-pool-1007-0257-f50a-58255159`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 9490, 'eta': '4h1m', 'grad': 0.0262, 'ce': 2.641}.
- f5: `mose-ffn-f5-r1-p1-1x8-pool-1007-0257-2198-70783152`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 9480, 'eta': '4h2m', 'grad': 0.0215, 'ce': 2.633}.
- a1: `mose-attn-a1-r1-p1-1x8-pool-1007-0128-5292-1-78ab4`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 10960, 'eta': '2h52m', 'grad': 0.0223, 'ce': 2.61}.
- a2: `mose-attn-a2-r1-p1-1x8-pool-1007-0128-b35f-2-1cf16`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 10990, 'eta': '2h50m', 'grad': 0.0522, 'ce': 2.633}.
- c0s2: `mose-r512-c0s2-r1-p1-1x8-pool-1006-2106-ea9c-f2516`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_210958-olmo3-1b-r512matched-c0-seed12537-data34521-fp8-30b'].
- c6s2: `mose-r512-c6s2-r1-p1-1x8-pool-1006-2107-3d78-52e71`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_211039-olmo3-1b-r512matched-c6-seed12537-data34521-fp8-30b'].
- c0s3: `mose-r512-c0s3-r1-p1-1x8-pool-1006-2107-2f04-80461`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_211054-olmo3-1b-r512matched-c0-seed12538-data34521-fp8-30b'].
- c6s3: `mose-r512-c6s3-r1-p1-1x8-pool-1006-2108-31a7-fc86e`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_211118-olmo3-1b-r512matched-c6-seed12538-data34521-fp8-30b'].
- c2s2: `mose-r512-c2s2-r1-p1-1x8-pool-1006-2338-833a-1f47a`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 13150, 'eta': '1h', 'grad': 0.0183, 'ce': 2.551}.
- c2s3: `mose-r512-c2s3-r1-p1-1x8-pool-1006-2338-d226-b5503`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 12790, 'eta': '1h20m', 'grad': 0.0124, 'ce': 2.454}.

## 2026-10-07 11:26 HKT — monitoring

- p0: `mose-ug-p0-r1-p1-1x8-pool-1006-1123-7104-77485485`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_112904-olmo3-1b-ug-partial-p0-s0-p512-linear-down-rms-fp8-30b'].
- p4: `mose-recover-p4-r2-p1-1x8-pool-1006-1207-0d3-44098`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_120940-olmo3-1b-ug-partial-p4-s160-p368-linear-down-rms-fp8-30b'].
- p3: `mose-recover-p3-r3-p1-1x8-pool-1006-1228-c38-15484`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_123139-olmo3-1b-ug-partial-p3-s320-p224-linear-down-rms-fp8-30b'].
- p2qk: `mose-ug-p2qk-r1-p1-4x8-arch-1006-1347-8485-7-afb90`; Succeeded; {'active': 0, 'succeeded': 4, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_135053-olmo3-1b-ug-partial-p2qk-s480-p80-linear-qk-silu-down-rms-fp8-30b'].
- p1: `mose-recover-p1-r3-p1-1x8-pool-1006-1358-5ee-ff2d5`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_140134-olmo3-1b-ug-partial-p1-s560-p0-linear-down-rms-fp8-30b'].
- p0qk: `mose-ug-p0qk-r1-p1-1x8-pool-1006-1347-459e-3-74da9`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_140141-olmo3-1b-ug-partial-p0qk-s0-p512-linear-qk-silu-down-rms-fp8-30b'].
- p2: `mose-recover-p2-r3-p1-2x8-arch-1006-1409-7a8-a7469`; Succeeded; {'active': 0, 'succeeded': 2, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_141212-olmo3-1b-ug-partial-p2-s480-p80-linear-down-rms-fp8-30b'].
- p3qk: `mose-ug-p3qk-r1-p1-1x8-pool-1006-2111-2dcc-7-ec6b7`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_211408-olmo3-1b-ug-partial-p3qk-s320-p224-linear-qk-silu-down-rms-fp8-30b'].
- p4qk: `mose-ug-p4qk-r1-p1-1x8-pool-1006-2111-0416-8-5686b`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_211424-olmo3-1b-ug-partial-p4qk-s160-p368-linear-qk-silu-down-rms-fp8-30b'].
- f1: `mose-recover-f1-r2-p1-1x8-pool-1006-1207-08d-af98d`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_121016-olmo3-1b-ffn-budget-f1-ug384-down768-unshared-linear-rms-fp8-30b'].
- f2: `mose-recover-f2-r2-p1-1x8-pool-1006-1213-0e1-bdefa`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_121541-olmo3-1b-ffn-budget-f2-ug256-down1024-unshared-linear-rms-fp8-30b'].
- f3: `mose-recover-f3-r2-p1-1x8-pool-1006-1213-ccc-5d862`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_121614-olmo3-1b-ffn-budget-f3-ug480-down672-shared-linear-rms-fp8-30b'].
- f3qk: `mose-ffn-f3qk-r1-p1-4x8-arch-1006-1923-76a7-5350e`; Succeeded; {'active': 0, 'succeeded': 4, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_192653-olmo3-1b-ffn-budget-f3qk-ug480-down672-shared-qk-silu-ug-linear-down-rms-fp8-30b'].
- f2qk: `mose-ffn-f2qk-r1-p1-1x8-pool-1006-1922-88d3-ed4c5`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_200818-olmo3-1b-ffn-budget-f2qk-ug256-down1024-unshared-qk-silu-ug-linear-down-rms-fp8-30b'].
- f1qk: `mose-ffn-f1qk-r2-p1-1x8-arch-1006-1937-bfde-3c52a`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_193938-olmo3-1b-ffn-budget-f1qk-ug384-down768-unshared-qk-silu-ug-linear-down-rms-fp8-30b'].
- f4qk: `mose-ffn-f4qk-r1-p1-2x8-arch-1006-2300-6d41-46be5`; Succeeded; {'active': 0, 'succeeded': 2, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_230506-olmo3-1b-ffn-budget-f4qk-ug400-down816-shared-qk-silu-ug-linear-down-rms-fp8-30b'].
- f5qk: `mose-ffn-f5qk-r1-p1-2x8-arch-1006-2300-f278-d8613`; Succeeded; {'active': 0, 'succeeded': 2, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_230543-olmo3-1b-ffn-budget-f5qk-ug560-down528-shared-qk-silu-ug-linear-down-rms-fp8-30b'].
- f4: `mose-ffn-f4-r1-p1-1x8-pool-1007-0257-f50a-58255159`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 10070, 'eta': '3h32m', 'grad': 0.0204, 'ce': 2.604}.
- f5: `mose-ffn-f5-r1-p1-1x8-pool-1007-0257-2198-70783152`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 10060, 'eta': '3h33m', 'grad': 0.0169, 'ce': 2.61}.
- a1: `mose-attn-a1-r1-p1-1x8-pool-1007-0128-5292-1-78ab4`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 11520, 'eta': '2h24m', 'grad': 0.0275, 'ce': 2.63}.
- a2: `mose-attn-a2-r1-p1-1x8-pool-1007-0128-b35f-2-1cf16`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 11570, 'eta': '2h21m', 'grad': 0.0195, 'ce': 2.676}.
- c0s2: `mose-r512-c0s2-r1-p1-1x8-pool-1006-2106-ea9c-f2516`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_210958-olmo3-1b-r512matched-c0-seed12537-data34521-fp8-30b'].
- c6s2: `mose-r512-c6s2-r1-p1-1x8-pool-1006-2107-3d78-52e71`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_211039-olmo3-1b-r512matched-c6-seed12537-data34521-fp8-30b'].
- c0s3: `mose-r512-c0s3-r1-p1-1x8-pool-1006-2107-2f04-80461`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_211054-olmo3-1b-r512matched-c0-seed12538-data34521-fp8-30b'].
- c6s3: `mose-r512-c6s3-r1-p1-1x8-pool-1006-2108-31a7-fc86e`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_211118-olmo3-1b-r512matched-c6-seed12538-data34521-fp8-30b'].
- c2s2: `mose-r512-c2s2-r1-p1-1x8-pool-1006-2338-833a-1f47a`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 13740, 'eta': '29m', 'grad': 0.0132, 'ce': 2.412}.
- c2s3: `mose-r512-c2s3-r1-p1-1x8-pool-1006-2338-d226-b5503`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 13370, 'eta': '50m', 'grad': 0.0119, 'ce': 2.539}.

## 2026-10-07 11:56 HKT — monitoring

- p0: `mose-ug-p0-r1-p1-1x8-pool-1006-1123-7104-77485485`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_112904-olmo3-1b-ug-partial-p0-s0-p512-linear-down-rms-fp8-30b'].
- p4: `mose-recover-p4-r2-p1-1x8-pool-1006-1207-0d3-44098`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_120940-olmo3-1b-ug-partial-p4-s160-p368-linear-down-rms-fp8-30b'].
- p3: `mose-recover-p3-r3-p1-1x8-pool-1006-1228-c38-15484`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_123139-olmo3-1b-ug-partial-p3-s320-p224-linear-down-rms-fp8-30b'].
- p2qk: `mose-ug-p2qk-r1-p1-4x8-arch-1006-1347-8485-7-afb90`; Succeeded; {'active': 0, 'succeeded': 4, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_135053-olmo3-1b-ug-partial-p2qk-s480-p80-linear-qk-silu-down-rms-fp8-30b'].
- p1: `mose-recover-p1-r3-p1-1x8-pool-1006-1358-5ee-ff2d5`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_140134-olmo3-1b-ug-partial-p1-s560-p0-linear-down-rms-fp8-30b'].
- p0qk: `mose-ug-p0qk-r1-p1-1x8-pool-1006-1347-459e-3-74da9`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_140141-olmo3-1b-ug-partial-p0qk-s0-p512-linear-qk-silu-down-rms-fp8-30b'].
- p2: `mose-recover-p2-r3-p1-2x8-arch-1006-1409-7a8-a7469`; Succeeded; {'active': 0, 'succeeded': 2, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_141212-olmo3-1b-ug-partial-p2-s480-p80-linear-down-rms-fp8-30b'].
- p3qk: `mose-ug-p3qk-r1-p1-1x8-pool-1006-2111-2dcc-7-ec6b7`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_211408-olmo3-1b-ug-partial-p3qk-s320-p224-linear-qk-silu-down-rms-fp8-30b'].
- p4qk: `mose-ug-p4qk-r1-p1-1x8-pool-1006-2111-0416-8-5686b`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_211424-olmo3-1b-ug-partial-p4qk-s160-p368-linear-qk-silu-down-rms-fp8-30b'].
- f1: `mose-recover-f1-r2-p1-1x8-pool-1006-1207-08d-af98d`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_121016-olmo3-1b-ffn-budget-f1-ug384-down768-unshared-linear-rms-fp8-30b'].
- f2: `mose-recover-f2-r2-p1-1x8-pool-1006-1213-0e1-bdefa`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_121541-olmo3-1b-ffn-budget-f2-ug256-down1024-unshared-linear-rms-fp8-30b'].
- f3: `mose-recover-f3-r2-p1-1x8-pool-1006-1213-ccc-5d862`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_121614-olmo3-1b-ffn-budget-f3-ug480-down672-shared-linear-rms-fp8-30b'].
- f3qk: `mose-ffn-f3qk-r1-p1-4x8-arch-1006-1923-76a7-5350e`; Succeeded; {'active': 0, 'succeeded': 4, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_192653-olmo3-1b-ffn-budget-f3qk-ug480-down672-shared-qk-silu-ug-linear-down-rms-fp8-30b'].
- f2qk: `mose-ffn-f2qk-r1-p1-1x8-pool-1006-1922-88d3-ed4c5`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_200818-olmo3-1b-ffn-budget-f2qk-ug256-down1024-unshared-qk-silu-ug-linear-down-rms-fp8-30b'].
- f1qk: `mose-ffn-f1qk-r2-p1-1x8-arch-1006-1937-bfde-3c52a`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_193938-olmo3-1b-ffn-budget-f1qk-ug384-down768-unshared-qk-silu-ug-linear-down-rms-fp8-30b'].
- f4qk: `mose-ffn-f4qk-r1-p1-2x8-arch-1006-2300-6d41-46be5`; Succeeded; {'active': 0, 'succeeded': 2, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_230506-olmo3-1b-ffn-budget-f4qk-ug400-down816-shared-qk-silu-ug-linear-down-rms-fp8-30b'].
- f5qk: `mose-ffn-f5qk-r1-p1-2x8-arch-1006-2300-f278-d8613`; Succeeded; {'active': 0, 'succeeded': 2, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_230543-olmo3-1b-ffn-budget-f5qk-ug560-down528-shared-qk-silu-ug-linear-down-rms-fp8-30b'].
- f4: `mose-ffn-f4-r1-p1-1x8-pool-1007-0257-f50a-58255159`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 10680, 'eta': '3h2m', 'grad': 0.0283, 'ce': 2.592}.
- f5: `mose-ffn-f5-r1-p1-1x8-pool-1007-0257-2198-70783152`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 10670, 'eta': '3h2m', 'grad': 0.0206, 'ce': 2.525}.
- a1: `mose-attn-a1-r1-p1-1x8-pool-1007-0128-5292-1-78ab4`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 12110, 'eta': '1h53m', 'grad': 0.0184, 'ce': 2.615}.
- a2: `mose-attn-a2-r1-p1-1x8-pool-1007-0128-b35f-2-1cf16`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 12160, 'eta': '1h50m', 'grad': 0.0159, 'ce': 2.597}.
- c0s2: `mose-r512-c0s2-r1-p1-1x8-pool-1006-2106-ea9c-f2516`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_210958-olmo3-1b-r512matched-c0-seed12537-data34521-fp8-30b'].
- c6s2: `mose-r512-c6s2-r1-p1-1x8-pool-1006-2107-3d78-52e71`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_211039-olmo3-1b-r512matched-c6-seed12537-data34521-fp8-30b'].
- c0s3: `mose-r512-c0s3-r1-p1-1x8-pool-1006-2107-2f04-80461`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_211054-olmo3-1b-r512matched-c0-seed12538-data34521-fp8-30b'].
- c6s3: `mose-r512-c6s3-r1-p1-1x8-pool-1006-2108-31a7-fc86e`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_211118-olmo3-1b-r512matched-c6-seed12538-data34521-fp8-30b'].
- c2s2: `mose-r512-c2s2-r1-p1-1x8-pool-1006-2338-833a-1f47a`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 14300, 'eta': '1m', 'grad': 0.0112, 'ce': 2.498}.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_234140-olmo3-1b-r512matched-c2-seed12537-data34521-fp8-30b', '/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_234140-olmo3-1b-r512matched-c2-seed12537-data34521-fp8-30b/logs'].
- c2s3: `mose-r512-c2s3-r1-p1-1x8-pool-1006-2338-d226-b5503`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 13950, 'eta': '19m', 'grad': 0.0108, 'ce': 2.526}.

## 2026-10-07 12:26 HKT — monitoring

- p0: `mose-ug-p0-r1-p1-1x8-pool-1006-1123-7104-77485485`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_112904-olmo3-1b-ug-partial-p0-s0-p512-linear-down-rms-fp8-30b'].
- p4: `mose-recover-p4-r2-p1-1x8-pool-1006-1207-0d3-44098`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_120940-olmo3-1b-ug-partial-p4-s160-p368-linear-down-rms-fp8-30b'].
- p3: `mose-recover-p3-r3-p1-1x8-pool-1006-1228-c38-15484`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_123139-olmo3-1b-ug-partial-p3-s320-p224-linear-down-rms-fp8-30b'].
- p2qk: `mose-ug-p2qk-r1-p1-4x8-arch-1006-1347-8485-7-afb90`; Succeeded; {'active': 0, 'succeeded': 4, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_135053-olmo3-1b-ug-partial-p2qk-s480-p80-linear-qk-silu-down-rms-fp8-30b'].
- p1: `mose-recover-p1-r3-p1-1x8-pool-1006-1358-5ee-ff2d5`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_140134-olmo3-1b-ug-partial-p1-s560-p0-linear-down-rms-fp8-30b'].
- p0qk: `mose-ug-p0qk-r1-p1-1x8-pool-1006-1347-459e-3-74da9`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_140141-olmo3-1b-ug-partial-p0qk-s0-p512-linear-qk-silu-down-rms-fp8-30b'].
- p2: `mose-recover-p2-r3-p1-2x8-arch-1006-1409-7a8-a7469`; Succeeded; {'active': 0, 'succeeded': 2, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_141212-olmo3-1b-ug-partial-p2-s480-p80-linear-down-rms-fp8-30b'].
- p3qk: `mose-ug-p3qk-r1-p1-1x8-pool-1006-2111-2dcc-7-ec6b7`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_211408-olmo3-1b-ug-partial-p3qk-s320-p224-linear-qk-silu-down-rms-fp8-30b'].
- p4qk: `mose-ug-p4qk-r1-p1-1x8-pool-1006-2111-0416-8-5686b`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_211424-olmo3-1b-ug-partial-p4qk-s160-p368-linear-qk-silu-down-rms-fp8-30b'].
- f1: `mose-recover-f1-r2-p1-1x8-pool-1006-1207-08d-af98d`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_121016-olmo3-1b-ffn-budget-f1-ug384-down768-unshared-linear-rms-fp8-30b'].
- f2: `mose-recover-f2-r2-p1-1x8-pool-1006-1213-0e1-bdefa`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_121541-olmo3-1b-ffn-budget-f2-ug256-down1024-unshared-linear-rms-fp8-30b'].
- f3: `mose-recover-f3-r2-p1-1x8-pool-1006-1213-ccc-5d862`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_121614-olmo3-1b-ffn-budget-f3-ug480-down672-shared-linear-rms-fp8-30b'].
- f3qk: `mose-ffn-f3qk-r1-p1-4x8-arch-1006-1923-76a7-5350e`; Succeeded; {'active': 0, 'succeeded': 4, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_192653-olmo3-1b-ffn-budget-f3qk-ug480-down672-shared-qk-silu-ug-linear-down-rms-fp8-30b'].
- f2qk: `mose-ffn-f2qk-r1-p1-1x8-pool-1006-1922-88d3-ed4c5`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_200818-olmo3-1b-ffn-budget-f2qk-ug256-down1024-unshared-qk-silu-ug-linear-down-rms-fp8-30b'].
- f1qk: `mose-ffn-f1qk-r2-p1-1x8-arch-1006-1937-bfde-3c52a`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_193938-olmo3-1b-ffn-budget-f1qk-ug384-down768-unshared-qk-silu-ug-linear-down-rms-fp8-30b'].
- f4qk: `mose-ffn-f4qk-r1-p1-2x8-arch-1006-2300-6d41-46be5`; Succeeded; {'active': 0, 'succeeded': 2, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_230506-olmo3-1b-ffn-budget-f4qk-ug400-down816-shared-qk-silu-ug-linear-down-rms-fp8-30b'].
- f5qk: `mose-ffn-f5qk-r1-p1-2x8-arch-1006-2300-f278-d8613`; Succeeded; {'active': 0, 'succeeded': 2, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_230543-olmo3-1b-ffn-budget-f5qk-ug560-down528-shared-qk-silu-ug-linear-down-rms-fp8-30b'].
- f4: `mose-ffn-f4-r1-p1-1x8-pool-1007-0257-f50a-58255159`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 11280, 'eta': '2h32m', 'grad': 0.0203, 'ce': 2.627}.
- f5: `mose-ffn-f5-r1-p1-1x8-pool-1007-0257-2198-70783152`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 11270, 'eta': '2h32m', 'grad': 0.0304, 'ce': 2.537}.
- a1: `mose-attn-a1-r1-p1-1x8-pool-1007-0128-5292-1-78ab4`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 12680, 'eta': '1h24m', 'grad': 0.0166, 'ce': 2.496}.
- a2: `mose-attn-a2-r1-p1-1x8-pool-1007-0128-b35f-2-1cf16`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 12730, 'eta': '1h21m', 'grad': 0.0112, 'ce': 2.556}.
- c0s2: `mose-r512-c0s2-r1-p1-1x8-pool-1006-2106-ea9c-f2516`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_210958-olmo3-1b-r512matched-c0-seed12537-data34521-fp8-30b'].
- c6s2: `mose-r512-c6s2-r1-p1-1x8-pool-1006-2107-3d78-52e71`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_211039-olmo3-1b-r512matched-c6-seed12537-data34521-fp8-30b'].
- c0s3: `mose-r512-c0s3-r1-p1-1x8-pool-1006-2107-2f04-80461`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_211054-olmo3-1b-r512matched-c0-seed12538-data34521-fp8-30b'].
- c6s3: `mose-r512-c6s3-r1-p1-1x8-pool-1006-2108-31a7-fc86e`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_211118-olmo3-1b-r512matched-c6-seed12538-data34521-fp8-30b'].
- c2s2: `mose-r512-c2s2-r1-p1-1x8-pool-1006-2338-833a-1f47a`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_234140-olmo3-1b-r512matched-c2-seed12537-data34521-fp8-30b'].
- c2s3: `mose-r512-c2s3-r1-p1-1x8-pool-1006-2338-d226-b5503`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 14300, 'eta': '1m', 'grad': 0.0095, 'ce': 2.498}.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_234305-olmo3-1b-r512matched-c2-seed12538-data34521-fp8-30b', '/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_234305-olmo3-1b-r512matched-c2-seed12538-data34521-fp8-30b/logs'].

## 2026-10-07 12:56 HKT — monitoring

- p0: `mose-ug-p0-r1-p1-1x8-pool-1006-1123-7104-77485485`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_112904-olmo3-1b-ug-partial-p0-s0-p512-linear-down-rms-fp8-30b'].
- p4: `mose-recover-p4-r2-p1-1x8-pool-1006-1207-0d3-44098`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_120940-olmo3-1b-ug-partial-p4-s160-p368-linear-down-rms-fp8-30b'].
- p3: `mose-recover-p3-r3-p1-1x8-pool-1006-1228-c38-15484`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_123139-olmo3-1b-ug-partial-p3-s320-p224-linear-down-rms-fp8-30b'].
- p2qk: `mose-ug-p2qk-r1-p1-4x8-arch-1006-1347-8485-7-afb90`; Succeeded; {'active': 0, 'succeeded': 4, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_135053-olmo3-1b-ug-partial-p2qk-s480-p80-linear-qk-silu-down-rms-fp8-30b'].
- p1: `mose-recover-p1-r3-p1-1x8-pool-1006-1358-5ee-ff2d5`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_140134-olmo3-1b-ug-partial-p1-s560-p0-linear-down-rms-fp8-30b'].
- p0qk: `mose-ug-p0qk-r1-p1-1x8-pool-1006-1347-459e-3-74da9`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_140141-olmo3-1b-ug-partial-p0qk-s0-p512-linear-qk-silu-down-rms-fp8-30b'].
- p2: `mose-recover-p2-r3-p1-2x8-arch-1006-1409-7a8-a7469`; Succeeded; {'active': 0, 'succeeded': 2, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_141212-olmo3-1b-ug-partial-p2-s480-p80-linear-down-rms-fp8-30b'].
- p3qk: `mose-ug-p3qk-r1-p1-1x8-pool-1006-2111-2dcc-7-ec6b7`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_211408-olmo3-1b-ug-partial-p3qk-s320-p224-linear-qk-silu-down-rms-fp8-30b'].
- p4qk: `mose-ug-p4qk-r1-p1-1x8-pool-1006-2111-0416-8-5686b`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_211424-olmo3-1b-ug-partial-p4qk-s160-p368-linear-qk-silu-down-rms-fp8-30b'].
- f1: `mose-recover-f1-r2-p1-1x8-pool-1006-1207-08d-af98d`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_121016-olmo3-1b-ffn-budget-f1-ug384-down768-unshared-linear-rms-fp8-30b'].
- f2: `mose-recover-f2-r2-p1-1x8-pool-1006-1213-0e1-bdefa`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_121541-olmo3-1b-ffn-budget-f2-ug256-down1024-unshared-linear-rms-fp8-30b'].
- f3: `mose-recover-f3-r2-p1-1x8-pool-1006-1213-ccc-5d862`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_121614-olmo3-1b-ffn-budget-f3-ug480-down672-shared-linear-rms-fp8-30b'].
- f3qk: `mose-ffn-f3qk-r1-p1-4x8-arch-1006-1923-76a7-5350e`; Succeeded; {'active': 0, 'succeeded': 4, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_192653-olmo3-1b-ffn-budget-f3qk-ug480-down672-shared-qk-silu-ug-linear-down-rms-fp8-30b'].
- f2qk: `mose-ffn-f2qk-r1-p1-1x8-pool-1006-1922-88d3-ed4c5`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_200818-olmo3-1b-ffn-budget-f2qk-ug256-down1024-unshared-qk-silu-ug-linear-down-rms-fp8-30b'].
- f1qk: `mose-ffn-f1qk-r2-p1-1x8-arch-1006-1937-bfde-3c52a`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_193938-olmo3-1b-ffn-budget-f1qk-ug384-down768-unshared-qk-silu-ug-linear-down-rms-fp8-30b'].
- f4qk: `mose-ffn-f4qk-r1-p1-2x8-arch-1006-2300-6d41-46be5`; Succeeded; {'active': 0, 'succeeded': 2, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_230506-olmo3-1b-ffn-budget-f4qk-ug400-down816-shared-qk-silu-ug-linear-down-rms-fp8-30b'].
- f5qk: `mose-ffn-f5qk-r1-p1-2x8-arch-1006-2300-f278-d8613`; Succeeded; {'active': 0, 'succeeded': 2, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_230543-olmo3-1b-ffn-budget-f5qk-ug560-down528-shared-qk-silu-ug-linear-down-rms-fp8-30b'].
- f4: `mose-ffn-f4-r1-p1-1x8-pool-1007-0257-f50a-58255159`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 11890, 'eta': '2h1m', 'grad': 0.0247, 'ce': 2.582}.
- f5: `mose-ffn-f5-r1-p1-1x8-pool-1007-0257-2198-70783152`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 11870, 'eta': '2h2m', 'grad': 0.0208, 'ce': 2.566}.
- a1: `mose-attn-a1-r1-p1-1x8-pool-1007-0128-5292-1-78ab4`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 13260, 'eta': '54m', 'grad': 0.0121, 'ce': 2.521}.
- a2: `mose-attn-a2-r1-p1-1x8-pool-1007-0128-b35f-2-1cf16`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 13320, 'eta': '51m', 'grad': 0.0254, 'ce': 2.471}.
- c0s2: `mose-r512-c0s2-r1-p1-1x8-pool-1006-2106-ea9c-f2516`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_210958-olmo3-1b-r512matched-c0-seed12537-data34521-fp8-30b'].
- c6s2: `mose-r512-c6s2-r1-p1-1x8-pool-1006-2107-3d78-52e71`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_211039-olmo3-1b-r512matched-c6-seed12537-data34521-fp8-30b'].
- c0s3: `mose-r512-c0s3-r1-p1-1x8-pool-1006-2107-2f04-80461`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_211054-olmo3-1b-r512matched-c0-seed12538-data34521-fp8-30b'].
- c6s3: `mose-r512-c6s3-r1-p1-1x8-pool-1006-2108-31a7-fc86e`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_211118-olmo3-1b-r512matched-c6-seed12538-data34521-fp8-30b'].
- c2s2: `mose-r512-c2s2-r1-p1-1x8-pool-1006-2338-833a-1f47a`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_234140-olmo3-1b-r512matched-c2-seed12537-data34521-fp8-30b'].
- c2s3: `mose-r512-c2s3-r1-p1-1x8-pool-1006-2338-d226-b5503`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_234305-olmo3-1b-r512matched-c2-seed12538-data34521-fp8-30b'].

## 2026-10-07 13:26 HKT — monitoring

- p0: `mose-ug-p0-r1-p1-1x8-pool-1006-1123-7104-77485485`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_112904-olmo3-1b-ug-partial-p0-s0-p512-linear-down-rms-fp8-30b'].
- p4: `mose-recover-p4-r2-p1-1x8-pool-1006-1207-0d3-44098`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_120940-olmo3-1b-ug-partial-p4-s160-p368-linear-down-rms-fp8-30b'].
- p3: `mose-recover-p3-r3-p1-1x8-pool-1006-1228-c38-15484`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_123139-olmo3-1b-ug-partial-p3-s320-p224-linear-down-rms-fp8-30b'].
- p2qk: `mose-ug-p2qk-r1-p1-4x8-arch-1006-1347-8485-7-afb90`; Succeeded; {'active': 0, 'succeeded': 4, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_135053-olmo3-1b-ug-partial-p2qk-s480-p80-linear-qk-silu-down-rms-fp8-30b'].
- p1: `mose-recover-p1-r3-p1-1x8-pool-1006-1358-5ee-ff2d5`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_140134-olmo3-1b-ug-partial-p1-s560-p0-linear-down-rms-fp8-30b'].
- p0qk: `mose-ug-p0qk-r1-p1-1x8-pool-1006-1347-459e-3-74da9`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_140141-olmo3-1b-ug-partial-p0qk-s0-p512-linear-qk-silu-down-rms-fp8-30b'].
- p2: `mose-recover-p2-r3-p1-2x8-arch-1006-1409-7a8-a7469`; Succeeded; {'active': 0, 'succeeded': 2, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_141212-olmo3-1b-ug-partial-p2-s480-p80-linear-down-rms-fp8-30b'].
- p3qk: `mose-ug-p3qk-r1-p1-1x8-pool-1006-2111-2dcc-7-ec6b7`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_211408-olmo3-1b-ug-partial-p3qk-s320-p224-linear-qk-silu-down-rms-fp8-30b'].
- p4qk: `mose-ug-p4qk-r1-p1-1x8-pool-1006-2111-0416-8-5686b`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_211424-olmo3-1b-ug-partial-p4qk-s160-p368-linear-qk-silu-down-rms-fp8-30b'].
- f1: `mose-recover-f1-r2-p1-1x8-pool-1006-1207-08d-af98d`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_121016-olmo3-1b-ffn-budget-f1-ug384-down768-unshared-linear-rms-fp8-30b'].
- f2: `mose-recover-f2-r2-p1-1x8-pool-1006-1213-0e1-bdefa`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_121541-olmo3-1b-ffn-budget-f2-ug256-down1024-unshared-linear-rms-fp8-30b'].
- f3: `mose-recover-f3-r2-p1-1x8-pool-1006-1213-ccc-5d862`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_121614-olmo3-1b-ffn-budget-f3-ug480-down672-shared-linear-rms-fp8-30b'].
- f3qk: `mose-ffn-f3qk-r1-p1-4x8-arch-1006-1923-76a7-5350e`; Succeeded; {'active': 0, 'succeeded': 4, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_192653-olmo3-1b-ffn-budget-f3qk-ug480-down672-shared-qk-silu-ug-linear-down-rms-fp8-30b'].
- f2qk: `mose-ffn-f2qk-r1-p1-1x8-pool-1006-1922-88d3-ed4c5`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_200818-olmo3-1b-ffn-budget-f2qk-ug256-down1024-unshared-qk-silu-ug-linear-down-rms-fp8-30b'].
- f1qk: `mose-ffn-f1qk-r2-p1-1x8-arch-1006-1937-bfde-3c52a`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_193938-olmo3-1b-ffn-budget-f1qk-ug384-down768-unshared-qk-silu-ug-linear-down-rms-fp8-30b'].
- f4qk: `mose-ffn-f4qk-r1-p1-2x8-arch-1006-2300-6d41-46be5`; Succeeded; {'active': 0, 'succeeded': 2, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_230506-olmo3-1b-ffn-budget-f4qk-ug400-down816-shared-qk-silu-ug-linear-down-rms-fp8-30b'].
- f5qk: `mose-ffn-f5qk-r1-p1-2x8-arch-1006-2300-f278-d8613`; Succeeded; {'active': 0, 'succeeded': 2, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_230543-olmo3-1b-ffn-budget-f5qk-ug560-down528-shared-qk-silu-ug-linear-down-rms-fp8-30b'].
- f4: `mose-ffn-f4-r1-p1-1x8-pool-1007-0257-f50a-58255159`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 12490, 'eta': '1h31m', 'grad': 0.014, 'ce': 2.536}.
- f5: `mose-ffn-f5-r1-p1-1x8-pool-1007-0257-2198-70783152`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 12480, 'eta': '1h32m', 'grad': 0.0133, 'ce': 2.601}.
- a1: `mose-attn-a1-r1-p1-1x8-pool-1007-0128-5292-1-78ab4`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 13850, 'eta': '24m', 'grad': 0.0129, 'ce': 2.454}.
- a2: `mose-attn-a2-r1-p1-1x8-pool-1007-0128-b35f-2-1cf16`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 13910, 'eta': '21m', 'grad': 0.0098, 'ce': 2.534}.
- c0s2: `mose-r512-c0s2-r1-p1-1x8-pool-1006-2106-ea9c-f2516`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_210958-olmo3-1b-r512matched-c0-seed12537-data34521-fp8-30b'].
- c6s2: `mose-r512-c6s2-r1-p1-1x8-pool-1006-2107-3d78-52e71`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_211039-olmo3-1b-r512matched-c6-seed12537-data34521-fp8-30b'].
- c0s3: `mose-r512-c0s3-r1-p1-1x8-pool-1006-2107-2f04-80461`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_211054-olmo3-1b-r512matched-c0-seed12538-data34521-fp8-30b'].
- c6s3: `mose-r512-c6s3-r1-p1-1x8-pool-1006-2108-31a7-fc86e`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_211118-olmo3-1b-r512matched-c6-seed12538-data34521-fp8-30b'].
- c2s2: `mose-r512-c2s2-r1-p1-1x8-pool-1006-2338-833a-1f47a`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_234140-olmo3-1b-r512matched-c2-seed12537-data34521-fp8-30b'].
- c2s3: `mose-r512-c2s3-r1-p1-1x8-pool-1006-2338-d226-b5503`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_234305-olmo3-1b-r512matched-c2-seed12538-data34521-fp8-30b'].

## 2026-10-07 13:56 HKT — monitoring

- p0: `mose-ug-p0-r1-p1-1x8-pool-1006-1123-7104-77485485`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_112904-olmo3-1b-ug-partial-p0-s0-p512-linear-down-rms-fp8-30b'].
- p4: `mose-recover-p4-r2-p1-1x8-pool-1006-1207-0d3-44098`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_120940-olmo3-1b-ug-partial-p4-s160-p368-linear-down-rms-fp8-30b'].
- p3: `mose-recover-p3-r3-p1-1x8-pool-1006-1228-c38-15484`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_123139-olmo3-1b-ug-partial-p3-s320-p224-linear-down-rms-fp8-30b'].
- p2qk: `mose-ug-p2qk-r1-p1-4x8-arch-1006-1347-8485-7-afb90`; Succeeded; {'active': 0, 'succeeded': 4, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_135053-olmo3-1b-ug-partial-p2qk-s480-p80-linear-qk-silu-down-rms-fp8-30b'].
- p1: `mose-recover-p1-r3-p1-1x8-pool-1006-1358-5ee-ff2d5`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_140134-olmo3-1b-ug-partial-p1-s560-p0-linear-down-rms-fp8-30b'].
- p0qk: `mose-ug-p0qk-r1-p1-1x8-pool-1006-1347-459e-3-74da9`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_140141-olmo3-1b-ug-partial-p0qk-s0-p512-linear-qk-silu-down-rms-fp8-30b'].
- p2: `mose-recover-p2-r3-p1-2x8-arch-1006-1409-7a8-a7469`; Succeeded; {'active': 0, 'succeeded': 2, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_141212-olmo3-1b-ug-partial-p2-s480-p80-linear-down-rms-fp8-30b'].
- p3qk: `mose-ug-p3qk-r1-p1-1x8-pool-1006-2111-2dcc-7-ec6b7`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_211408-olmo3-1b-ug-partial-p3qk-s320-p224-linear-qk-silu-down-rms-fp8-30b'].
- p4qk: `mose-ug-p4qk-r1-p1-1x8-pool-1006-2111-0416-8-5686b`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_211424-olmo3-1b-ug-partial-p4qk-s160-p368-linear-qk-silu-down-rms-fp8-30b'].
- f1: `mose-recover-f1-r2-p1-1x8-pool-1006-1207-08d-af98d`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_121016-olmo3-1b-ffn-budget-f1-ug384-down768-unshared-linear-rms-fp8-30b'].
- f2: `mose-recover-f2-r2-p1-1x8-pool-1006-1213-0e1-bdefa`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_121541-olmo3-1b-ffn-budget-f2-ug256-down1024-unshared-linear-rms-fp8-30b'].
- f3: `mose-recover-f3-r2-p1-1x8-pool-1006-1213-ccc-5d862`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_121614-olmo3-1b-ffn-budget-f3-ug480-down672-shared-linear-rms-fp8-30b'].
- f3qk: `mose-ffn-f3qk-r1-p1-4x8-arch-1006-1923-76a7-5350e`; Succeeded; {'active': 0, 'succeeded': 4, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_192653-olmo3-1b-ffn-budget-f3qk-ug480-down672-shared-qk-silu-ug-linear-down-rms-fp8-30b'].
- f2qk: `mose-ffn-f2qk-r1-p1-1x8-pool-1006-1922-88d3-ed4c5`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_200818-olmo3-1b-ffn-budget-f2qk-ug256-down1024-unshared-qk-silu-ug-linear-down-rms-fp8-30b'].
- f1qk: `mose-ffn-f1qk-r2-p1-1x8-arch-1006-1937-bfde-3c52a`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_193938-olmo3-1b-ffn-budget-f1qk-ug384-down768-unshared-qk-silu-ug-linear-down-rms-fp8-30b'].
- f4qk: `mose-ffn-f4qk-r1-p1-2x8-arch-1006-2300-6d41-46be5`; Succeeded; {'active': 0, 'succeeded': 2, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_230506-olmo3-1b-ffn-budget-f4qk-ug400-down816-shared-qk-silu-ug-linear-down-rms-fp8-30b'].
- f5qk: `mose-ffn-f5qk-r1-p1-2x8-arch-1006-2300-f278-d8613`; Succeeded; {'active': 0, 'succeeded': 2, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_230543-olmo3-1b-ffn-budget-f5qk-ug560-down528-shared-qk-silu-ug-linear-down-rms-fp8-30b'].
- f4: `mose-ffn-f4-r1-p1-1x8-pool-1007-0257-f50a-58255159`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 13050, 'eta': '1h3m', 'grad': 0.017, 'ce': 2.559}.
- f5: `mose-ffn-f5-r1-p1-1x8-pool-1007-0257-2198-70783152`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 13030, 'eta': '1h4m', 'grad': 0.0191, 'ce': 2.581}.
- a1: `mose-attn-a1-r1-p1-1x8-pool-1007-0128-5292-1-78ab4`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 14300, 'eta': '1m', 'grad': 0.0077, 'ce': 2.522}.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261007_013128-olmo3-1b-attn-budget-a1-r832-ug336-down352-unshared-qk-silu-down-rms-fp8-30b', '/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261007_013128-olmo3-1b-attn-budget-a1-r832-ug336-down352-unshared-qk-silu-down-rms-fp8-30b/logs'].
- a2: `mose-attn-a2-r1-p1-1x8-pool-1007-0128-b35f-2-1cf16`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 14300, 'eta': '1m', 'grad': 0.0074, 'ce': 2.522}.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261007_013127-olmo3-1b-attn-budget-a2-r832-ug320-down448-shared-qk-silu-down-rms-fp8-30b', '/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261007_013127-olmo3-1b-attn-budget-a2-r832-ug320-down448-shared-qk-silu-down-rms-fp8-30b/logs'].
- c0s2: `mose-r512-c0s2-r1-p1-1x8-pool-1006-2106-ea9c-f2516`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_210958-olmo3-1b-r512matched-c0-seed12537-data34521-fp8-30b'].
- c6s2: `mose-r512-c6s2-r1-p1-1x8-pool-1006-2107-3d78-52e71`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_211039-olmo3-1b-r512matched-c6-seed12537-data34521-fp8-30b'].
- c0s3: `mose-r512-c0s3-r1-p1-1x8-pool-1006-2107-2f04-80461`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_211054-olmo3-1b-r512matched-c0-seed12538-data34521-fp8-30b'].
- c6s3: `mose-r512-c6s3-r1-p1-1x8-pool-1006-2108-31a7-fc86e`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_211118-olmo3-1b-r512matched-c6-seed12538-data34521-fp8-30b'].
- c2s2: `mose-r512-c2s2-r1-p1-1x8-pool-1006-2338-833a-1f47a`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_234140-olmo3-1b-r512matched-c2-seed12537-data34521-fp8-30b'].
- c2s3: `mose-r512-c2s3-r1-p1-1x8-pool-1006-2338-d226-b5503`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_234305-olmo3-1b-r512matched-c2-seed12538-data34521-fp8-30b'].

## 2026-10-07 14:26 HKT — monitoring

- p0: `mose-ug-p0-r1-p1-1x8-pool-1006-1123-7104-77485485`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_112904-olmo3-1b-ug-partial-p0-s0-p512-linear-down-rms-fp8-30b'].
- p4: `mose-recover-p4-r2-p1-1x8-pool-1006-1207-0d3-44098`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_120940-olmo3-1b-ug-partial-p4-s160-p368-linear-down-rms-fp8-30b'].
- p3: `mose-recover-p3-r3-p1-1x8-pool-1006-1228-c38-15484`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_123139-olmo3-1b-ug-partial-p3-s320-p224-linear-down-rms-fp8-30b'].
- p2qk: `mose-ug-p2qk-r1-p1-4x8-arch-1006-1347-8485-7-afb90`; Succeeded; {'active': 0, 'succeeded': 4, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_135053-olmo3-1b-ug-partial-p2qk-s480-p80-linear-qk-silu-down-rms-fp8-30b'].
- p1: `mose-recover-p1-r3-p1-1x8-pool-1006-1358-5ee-ff2d5`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_140134-olmo3-1b-ug-partial-p1-s560-p0-linear-down-rms-fp8-30b'].
- p0qk: `mose-ug-p0qk-r1-p1-1x8-pool-1006-1347-459e-3-74da9`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_140141-olmo3-1b-ug-partial-p0qk-s0-p512-linear-qk-silu-down-rms-fp8-30b'].
- p2: `mose-recover-p2-r3-p1-2x8-arch-1006-1409-7a8-a7469`; Succeeded; {'active': 0, 'succeeded': 2, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_141212-olmo3-1b-ug-partial-p2-s480-p80-linear-down-rms-fp8-30b'].
- p3qk: `mose-ug-p3qk-r1-p1-1x8-pool-1006-2111-2dcc-7-ec6b7`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_211408-olmo3-1b-ug-partial-p3qk-s320-p224-linear-qk-silu-down-rms-fp8-30b'].
- p4qk: `mose-ug-p4qk-r1-p1-1x8-pool-1006-2111-0416-8-5686b`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_211424-olmo3-1b-ug-partial-p4qk-s160-p368-linear-qk-silu-down-rms-fp8-30b'].
- f1: `mose-recover-f1-r2-p1-1x8-pool-1006-1207-08d-af98d`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_121016-olmo3-1b-ffn-budget-f1-ug384-down768-unshared-linear-rms-fp8-30b'].
- f2: `mose-recover-f2-r2-p1-1x8-pool-1006-1213-0e1-bdefa`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_121541-olmo3-1b-ffn-budget-f2-ug256-down1024-unshared-linear-rms-fp8-30b'].
- f3: `mose-recover-f3-r2-p1-1x8-pool-1006-1213-ccc-5d862`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_121614-olmo3-1b-ffn-budget-f3-ug480-down672-shared-linear-rms-fp8-30b'].
- f3qk: `mose-ffn-f3qk-r1-p1-4x8-arch-1006-1923-76a7-5350e`; Succeeded; {'active': 0, 'succeeded': 4, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_192653-olmo3-1b-ffn-budget-f3qk-ug480-down672-shared-qk-silu-ug-linear-down-rms-fp8-30b'].
- f2qk: `mose-ffn-f2qk-r1-p1-1x8-pool-1006-1922-88d3-ed4c5`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_200818-olmo3-1b-ffn-budget-f2qk-ug256-down1024-unshared-qk-silu-ug-linear-down-rms-fp8-30b'].
- f1qk: `mose-ffn-f1qk-r2-p1-1x8-arch-1006-1937-bfde-3c52a`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_193938-olmo3-1b-ffn-budget-f1qk-ug384-down768-unshared-qk-silu-ug-linear-down-rms-fp8-30b'].
- f4qk: `mose-ffn-f4qk-r1-p1-2x8-arch-1006-2300-6d41-46be5`; Succeeded; {'active': 0, 'succeeded': 2, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_230506-olmo3-1b-ffn-budget-f4qk-ug400-down816-shared-qk-silu-ug-linear-down-rms-fp8-30b'].
- f5qk: `mose-ffn-f5qk-r1-p1-2x8-arch-1006-2300-f278-d8613`; Succeeded; {'active': 0, 'succeeded': 2, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_230543-olmo3-1b-ffn-budget-f5qk-ug560-down528-shared-qk-silu-ug-linear-down-rms-fp8-30b'].
- f4: `mose-ffn-f4-r1-p1-1x8-pool-1007-0257-f50a-58255159`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 13650, 'eta': '33m', 'grad': 0.0098, 'ce': 2.529}.
- f5: `mose-ffn-f5-r1-p1-1x8-pool-1007-0257-2198-70783152`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 13640, 'eta': '34m', 'grad': 0.0119, 'ce': 2.476}.
- a1: `mose-attn-a1-r1-p1-1x8-pool-1007-0128-5292-1-78ab4`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 14300, 'eta': '1m', 'grad': 0.0077, 'ce': 2.522}.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261007_013128-olmo3-1b-attn-budget-a1-r832-ug336-down352-unshared-qk-silu-down-rms-fp8-30b', '/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261007_013128-olmo3-1b-attn-budget-a1-r832-ug336-down352-unshared-qk-silu-down-rms-fp8-30b/logs'].
- a2: `mose-attn-a2-r1-p1-1x8-pool-1007-0128-b35f-2-1cf16`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 14300, 'eta': '1m', 'grad': 0.0074, 'ce': 2.522}.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261007_013127-olmo3-1b-attn-budget-a2-r832-ug320-down448-shared-qk-silu-down-rms-fp8-30b', '/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261007_013127-olmo3-1b-attn-budget-a2-r832-ug320-down448-shared-qk-silu-down-rms-fp8-30b/logs'].
- c0s2: `mose-r512-c0s2-r1-p1-1x8-pool-1006-2106-ea9c-f2516`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_210958-olmo3-1b-r512matched-c0-seed12537-data34521-fp8-30b'].
- c6s2: `mose-r512-c6s2-r1-p1-1x8-pool-1006-2107-3d78-52e71`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_211039-olmo3-1b-r512matched-c6-seed12537-data34521-fp8-30b'].
- c0s3: `mose-r512-c0s3-r1-p1-1x8-pool-1006-2107-2f04-80461`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_211054-olmo3-1b-r512matched-c0-seed12538-data34521-fp8-30b'].
- c6s3: `mose-r512-c6s3-r1-p1-1x8-pool-1006-2108-31a7-fc86e`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_211118-olmo3-1b-r512matched-c6-seed12538-data34521-fp8-30b'].
- c2s2: `mose-r512-c2s2-r1-p1-1x8-pool-1006-2338-833a-1f47a`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_234140-olmo3-1b-r512matched-c2-seed12537-data34521-fp8-30b'].
- c2s3: `mose-r512-c2s3-r1-p1-1x8-pool-1006-2338-d226-b5503`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_234305-olmo3-1b-r512matched-c2-seed12538-data34521-fp8-30b'].

## 2026-10-07 14:56 HKT — monitoring

- p0: `mose-ug-p0-r1-p1-1x8-pool-1006-1123-7104-77485485`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_112904-olmo3-1b-ug-partial-p0-s0-p512-linear-down-rms-fp8-30b'].
- p4: `mose-recover-p4-r2-p1-1x8-pool-1006-1207-0d3-44098`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_120940-olmo3-1b-ug-partial-p4-s160-p368-linear-down-rms-fp8-30b'].
- p3: `mose-recover-p3-r3-p1-1x8-pool-1006-1228-c38-15484`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_123139-olmo3-1b-ug-partial-p3-s320-p224-linear-down-rms-fp8-30b'].
- p2qk: `mose-ug-p2qk-r1-p1-4x8-arch-1006-1347-8485-7-afb90`; Succeeded; {'active': 0, 'succeeded': 4, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_135053-olmo3-1b-ug-partial-p2qk-s480-p80-linear-qk-silu-down-rms-fp8-30b'].
- p1: `mose-recover-p1-r3-p1-1x8-pool-1006-1358-5ee-ff2d5`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_140134-olmo3-1b-ug-partial-p1-s560-p0-linear-down-rms-fp8-30b'].
- p0qk: `mose-ug-p0qk-r1-p1-1x8-pool-1006-1347-459e-3-74da9`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_140141-olmo3-1b-ug-partial-p0qk-s0-p512-linear-qk-silu-down-rms-fp8-30b'].
- p2: `mose-recover-p2-r3-p1-2x8-arch-1006-1409-7a8-a7469`; Succeeded; {'active': 0, 'succeeded': 2, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_141212-olmo3-1b-ug-partial-p2-s480-p80-linear-down-rms-fp8-30b'].
- p3qk: `mose-ug-p3qk-r1-p1-1x8-pool-1006-2111-2dcc-7-ec6b7`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_211408-olmo3-1b-ug-partial-p3qk-s320-p224-linear-qk-silu-down-rms-fp8-30b'].
- p4qk: `mose-ug-p4qk-r1-p1-1x8-pool-1006-2111-0416-8-5686b`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_211424-olmo3-1b-ug-partial-p4qk-s160-p368-linear-qk-silu-down-rms-fp8-30b'].
- f1: `mose-recover-f1-r2-p1-1x8-pool-1006-1207-08d-af98d`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_121016-olmo3-1b-ffn-budget-f1-ug384-down768-unshared-linear-rms-fp8-30b'].
- f2: `mose-recover-f2-r2-p1-1x8-pool-1006-1213-0e1-bdefa`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_121541-olmo3-1b-ffn-budget-f2-ug256-down1024-unshared-linear-rms-fp8-30b'].
- f3: `mose-recover-f3-r2-p1-1x8-pool-1006-1213-ccc-5d862`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_121614-olmo3-1b-ffn-budget-f3-ug480-down672-shared-linear-rms-fp8-30b'].
- f3qk: `mose-ffn-f3qk-r1-p1-4x8-arch-1006-1923-76a7-5350e`; Succeeded; {'active': 0, 'succeeded': 4, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_192653-olmo3-1b-ffn-budget-f3qk-ug480-down672-shared-qk-silu-ug-linear-down-rms-fp8-30b'].
- f2qk: `mose-ffn-f2qk-r1-p1-1x8-pool-1006-1922-88d3-ed4c5`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_200818-olmo3-1b-ffn-budget-f2qk-ug256-down1024-unshared-qk-silu-ug-linear-down-rms-fp8-30b'].
- f1qk: `mose-ffn-f1qk-r2-p1-1x8-arch-1006-1937-bfde-3c52a`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_193938-olmo3-1b-ffn-budget-f1qk-ug384-down768-unshared-qk-silu-ug-linear-down-rms-fp8-30b'].
- f4qk: `mose-ffn-f4qk-r1-p1-2x8-arch-1006-2300-6d41-46be5`; Succeeded; {'active': 0, 'succeeded': 2, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_230506-olmo3-1b-ffn-budget-f4qk-ug400-down816-shared-qk-silu-ug-linear-down-rms-fp8-30b'].
- f5qk: `mose-ffn-f5qk-r1-p1-2x8-arch-1006-2300-f278-d8613`; Succeeded; {'active': 0, 'succeeded': 2, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_230543-olmo3-1b-ffn-budget-f5qk-ug560-down528-shared-qk-silu-ug-linear-down-rms-fp8-30b'].
- f4: `mose-ffn-f4-r1-p1-1x8-pool-1007-0257-f50a-58255159`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 14250, 'eta': '3m', 'grad': 0.0081, 'ce': 2.501}.
- f5: `mose-ffn-f5-r1-p1-1x8-pool-1007-0257-2198-70783152`; Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 14240, 'eta': '4m', 'grad': 0.0081, 'ce': 2.423}.
- a1: `mose-attn-a1-r1-p1-1x8-pool-1007-0128-5292-1-78ab4`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 14300, 'eta': '1m', 'grad': 0.0077, 'ce': 2.522}.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261007_013128-olmo3-1b-attn-budget-a1-r832-ug336-down352-unshared-qk-silu-down-rms-fp8-30b', '/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261007_013128-olmo3-1b-attn-budget-a1-r832-ug336-down352-unshared-qk-silu-down-rms-fp8-30b/logs'].
- a2: `mose-attn-a2-r1-p1-1x8-pool-1007-0128-b35f-2-1cf16`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; {'step': 14300, 'eta': '1m', 'grad': 0.0074, 'ce': 2.522}.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261007_013127-olmo3-1b-attn-budget-a2-r832-ug320-down448-shared-qk-silu-down-rms-fp8-30b', '/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261007_013127-olmo3-1b-attn-budget-a2-r832-ug320-down448-shared-qk-silu-down-rms-fp8-30b/logs'].
- c0s2: `mose-r512-c0s2-r1-p1-1x8-pool-1006-2106-ea9c-f2516`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_210958-olmo3-1b-r512matched-c0-seed12537-data34521-fp8-30b'].
- c6s2: `mose-r512-c6s2-r1-p1-1x8-pool-1006-2107-3d78-52e71`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_211039-olmo3-1b-r512matched-c6-seed12537-data34521-fp8-30b'].
- c0s3: `mose-r512-c0s3-r1-p1-1x8-pool-1006-2107-2f04-80461`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_211054-olmo3-1b-r512matched-c0-seed12538-data34521-fp8-30b'].
- c6s3: `mose-r512-c6s3-r1-p1-1x8-pool-1006-2108-31a7-fc86e`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_211118-olmo3-1b-r512matched-c6-seed12538-data34521-fp8-30b'].
- c2s2: `mose-r512-c2s2-r1-p1-1x8-pool-1006-2338-833a-1f47a`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_234140-olmo3-1b-r512matched-c2-seed12537-data34521-fp8-30b'].
- c2s3: `mose-r512-c2s3-r1-p1-1x8-pool-1006-2338-d226-b5503`; Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; None.
  Exact W&B segments: ['/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_234305-olmo3-1b-r512matched-c2-seed12538-data34521-fp8-30b'].
