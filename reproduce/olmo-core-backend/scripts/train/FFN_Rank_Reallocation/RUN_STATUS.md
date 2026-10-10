# FFN rank reallocation runs

2026-10-06: complement the UG-sharing study by moving the fixed rank512 parameter budget from UG to Down. User allows concurrent pool1x8 low-priority experiments. Three configs prepared; no task submitted yet; local validation in progress.

Local validation passed: actual meta-module FFN counts15,728,640 per block and full-model797,050,880 for all3; unchanged attention checked; rank-override rejection checked. All3 full training configs, seeds, FP8, schedule, retention and batch divisibility validated. Python/shell syntax passed.

Submitted F1: `mose-ffn-f1-r1-p1-1x8-pool-1006-1135-4cae-99748743`, pool 1x8, priority1; script `/mnt/shared-storage-gpfs2/intern-pretrain-shared02/shijiayang-p/archspace-MoSE/reproduce/olmo-core-backend/scripts/train/FFN_Rank_Reallocation/run_f1.sh`.

Submitted F2: `mose-ffn-f2-r1-p1-1x8-pool-1006-1135-6c39-5628140`, pool 1x8, priority1; script `/mnt/shared-storage-gpfs2/intern-pretrain-shared02/shijiayang-p/archspace-MoSE/reproduce/olmo-core-backend/scripts/train/FFN_Rank_Reallocation/run_f2.sh`.

Submitted F3: `mose-ffn-f3-r1-p1-1x8-pool-1006-1135-0526-12030032`, pool 1x8, priority1; script `/mnt/shared-storage-gpfs2/intern-pretrain-shared02/shijiayang-p/archspace-MoSE/reproduce/olmo-core-backend/scripts/train/FFN_Rank_Reallocation/run_f3.sh`.

Unified monitoring covers5 UG-sharing and3 FFN-reallocation jobs. Replaced only the local monitoring process to load the expanded registry; all GPU jobs remain running. Next regular check11:57 HKT, then every1800 seconds.

Generalized offline analysis verified against completed Dense/S1/S2/S3 histories: all prior full/tail/final CE reproduce to1e-12. Supports common-step anomaly sensitivity and identical11-source validation.

Submission audit: all active variants have unique scheduler IDs, config launchers and run paths; allpool accounting confirmed llmarchitecture_gpu_pool. No duplicate live attempts. Code remains on arch/MoSE-no-kvcache.

11:39 HKT: all3pool jobs Running active1/failed0, mesh(dp_replicate1,dp_shard8); first-step verification due at scheduled11:57 check.

## 2026-10-06 11:57–12:05 HKT — failure diagnosis and correction

P1/P4/F1/F3 have explicit Disk quota exceeded in GCC/Inductor writes. F3 reports /mnt/shared-storage-user/jitao/tmp_data/cc*.s. F2 reports CUDA driver error: device kernel image is invalid; shared-cache corruption is a hypothesis, not a verified root cause. SIGTERM lines are the distributed shutdown after the primary failures.
Launchers now create fresh per-job/per-node disposable TMPDIR, TorchInductor, Triton and CUDA caches under /tmp before invoking hceph/Python. Setup and bash syntax tested locally. Work/data/checkpoint paths and all architecture/hyperparameters are preserved. Only verified Failed/active0 attempts are synchronized and replaced; healthy jobs continue.
Arch restart prediction:4x8 and2x8 unavailable (only1 replica schedulable); authorized replacements use pool1x8 priority1.

Recovery f1: old `mose-ffn-f1-r1-p1-1x8-pool-1006-1135-4cae-99748743` verified Failed/active0. Exact old W&B segment Finished syncing. Same run/checkpoint/config; only disposable compiler/temp caches redirected to job-local /tmp. New pool1x8 priority1 attempt follows.

Submitted F1: `mose-recover-f1-r2-p1-1x8-pool-1006-1207-08d-af98d`, pool 1x8, priority1; script `/mnt/shared-storage-gpfs2/intern-pretrain-shared02/shijiayang-p/archspace-MoSE/reproduce/olmo-core-backend/scripts/train/FFN_Rank_Reallocation/run_f1.sh`.

Recovery f2: old `mose-ffn-f2-r1-p1-1x8-pool-1006-1135-6c39-5628140` verified Failed/active0. Exact old W&B synchronization state: failed. Same run/checkpoint/config; only disposable compiler/temp caches redirected to job-local /tmp. New pool1x8 priority1 attempt follows.

Submitted F2: `mose-recover-f2-r2-p1-1x8-pool-1006-1213-0e1-bdefa`, pool 1x8, priority1; script `/mnt/shared-storage-gpfs2/intern-pretrain-shared02/shijiayang-p/archspace-MoSE/reproduce/olmo-core-backend/scripts/train/FFN_Rank_Reallocation/run_f2.sh`.

Recovery f3: old `mose-ffn-f3-r1-p1-1x8-pool-1006-1135-0526-12030032` verified Failed/active0. Exact old W&B synchronization state: started. Same run/checkpoint/config; only disposable compiler/temp caches redirected to job-local /tmp. New pool1x8 priority1 attempt follows.

Submitted F3: `mose-recover-f3-r2-p1-1x8-pool-1006-1213-ccc-5d862`, pool 1x8, priority1; script `/mnt/shared-storage-gpfs2/intern-pretrain-shared02/shijiayang-p/archspace-MoSE/reproduce/olmo-core-backend/scripts/train/FFN_Rank_Reallocation/run_f3.sh`.

12:14 HKT recovery verification: P1/P4/F1 resumed step0, passed dry-run and emitted finite train CE. F2/F3 replacements Running active1 on pool1x8, fresh job-local caches and (dp_replicate1,dp_shard8) verified; first CE pending at this check. Old F2 upload encountered transaction-log unexpected EOF; original record preserved, completion is not claimed. Same hyperparameters and outputs retained.

12:27 HKT: F1/F2/F3 replacements have finite train CE and Running/active1. Old F2 and F3 offline transaction logs are truncated (unexpected EOF / next-record EOF); both upload invocations are recorded as failed, originals preserved. Neither old attempt had periodic train CE; new attempts resumed step0. No blind upload retries.

Submitted F3QK: `mose-ffn-f3qk-r1-p1-4x8-arch-1006-1923-76a7-5350e`, arch 4x8, priority1; script `/mnt/shared-storage-gpfs2/intern-pretrain-shared02/shijiayang-p/archspace-MoSE/reproduce/olmo-core-backend/scripts/train/FFN_Rank_Reallocation/run_f3qk.sh`.

Submitted F1QK: `mose-ffn-f1qk-r1-p1-1x8-pool-1006-1925-79f8-ca40a`, pool 1x8, priority1; script `/mnt/shared-storage-gpfs2/intern-pretrain-shared02/shijiayang-p/archspace-MoSE/reproduce/olmo-core-backend/scripts/train/FFN_Rank_Reallocation/run_f1qk.sh`.

Submitted F2QK: `mose-ffn-f2qk-r1-p1-1x8-pool-1006-1922-88d3-ed4c5`, pool 1x8, priority1; script `/mnt/shared-storage-gpfs2/intern-pretrain-shared02/shijiayang-p/archspace-MoSE/reproduce/olmo-core-backend/scripts/train/FFN_Rank_Reallocation/run_f2qk.sh`.

19:28 HKT startup check: F3qk Running/active4 on arch32, mesh(4,8), GPFS2 checkpoint preflight passed (3.7TiB filesystem available;16MiB write probe succeeded), exact W&B segment recorded; finite training CE still awaiting verification. After this allocation, arch16 predicted unavailable (onlyone of two replicas schedulable), so F1qk/F2qk were submitted to pool1x8 priority1 and are Inqueue/active0. Queueing is not stopped and does not authorize duplicate attempts. Existing first ten configs remain unchanged; unified monitor dynamically includes all thirteen.

19:32 HKT: F3qk actual training verified through step297, finite CE4.582/4.365/4.273, no errors/OOM; stage1/step0 saved successfully at19:26:54.526. Startup verification now true for step0 (asynchronous save uses “Checkpoint for step0 saved successfully”, rather than generic “Checkpoint saved”). F1qk/F2qk remain Inqueue/active0.

1006-1937 F1qk old pool attempt verified Stopped/active0; it was Inqueue with all counterszero and never produced a W&B segment (authorized log retrieval returned no replica data). Synchronization skipped for this never-started attempt. Arch had one available8-GPU node but no immediately available16; use8-card fallback allowed by existing preference, preserve all recipe/model/globalbatch settings. New unique attempt mose-ffn-f1qk-r2-p1-1x8-arch-1006-1937-bfde. No healthy job was moved.

Submitted F1QK: `mose-ffn-f1qk-r2-p1-1x8-arch-1006-1937-bfde-3c52a`, arch 1x8, priority1; script `/mnt/shared-storage-gpfs2/intern-pretrain-shared02/shijiayang-p/archspace-MoSE/reproduce/olmo-core-backend/scripts/train/FFN_Rank_Reallocation/run_f1qk.sh`.

19:53 HKT: F3qk arch32 training through1930 (CE2.966), F1qk arch8 through240 (CE4.775); both finite with expected active replicas. F2qk still Inqueue/active0 onpool8. Existing originaleight unfinished jobs allhealthy at6980–10060; two originalconfigs completed and uploaded. Uniform30min monitor81192 covers13 total configurations.

20:23 HKT: All three new combinations now actively train: F3qk arch32 step4280 CE2.737; F1qk arch8 step830 CE3.185; F2qk pool8 step280 CE4.295. Finite training verified for all, no restart of queued/live attempts. Total thirteen configs: two completed/uploaded, eleven Running. Native SwiGLU/Down RMSNorm and exact797,050,880 budgets maintained. Unified30min monitor remains live.

## 2026-10-06 22:56 HKT — F3qk completion verified

F3qk Succeeded/active0/succeeded4; final14306 checkpoint saved at22:34:32, Training complete22:34:35. Completion proof saved. Full-precision CE comparison with archived C0/C2/C6 is prepared; ranking remains pending.

Submitted F4QK: `mose-ffn-f4qk-r1-p1-2x8-arch-1006-2300-6d41-46be5`, arch 2x8, priority1; script `/mnt/shared-storage-gpfs2/intern-pretrain-shared02/shijiayang-p/archspace-MoSE/reproduce/olmo-core-backend/scripts/train/FFN_Rank_Reallocation/run_f4qk.sh`.

Submitted F5QK: `mose-ffn-f5qk-r1-p1-2x8-arch-1006-2300-f278-d8613`, arch 2x8, priority1; script `/mnt/shared-storage-gpfs2/intern-pretrain-shared02/shijiayang-p/archspace-MoSE/reproduce/olmo-core-backend/scripts/train/FFN_Rank_Reallocation/run_f5qk.sh`.

## 2026-10-07 00:26 HKT — F1/F2/F3 completed

All three current r2 attempts Succeeded/active0/succeeded1, full14306 with final checkpoints saved. Exact full current segments selected; early failed r1 histories are not mixed into fresh complete runs. Full CE comparison prepared; uploads pending verification.

## 2026-10-07 01:27 HKT — complete primary FFN quality analysis

F1/F2/F3 full30B histories and original C0/C2/C6 recomputed with one common anomaly mask. Final1000 mean: F1=2.482142080, F2=2.498585725, F3=2.481781711; C6=2.474762967 remains best. F3qk=2.477421343 improves F3 by0.004360369, with32/8GPU allocation difference retained. Analysis job Succeeded/active0, JSON checksum verified. See ../UG_Partial_Sharing/FFN_PRIMARY_WITH_REFERENCES_RESULTS.md.

2026-10-07 02:47 HKT: F4/F5 Identity-attention controls prepared locally. Existing `_common.VARIANTS` and prior all-five QK recipe/budget validation already cover these exact primary models. Only new entrypoints/launchers added; Python compile and bash-n passed. Await fresh capacity before submission; no job created yet.

2026-10-07 02:57 HKT: aligned-clean-history test for F4/F5 QK contrasts passed1test10.26s. New launchers syntax valid. Fresh arch2x8/4x8 predictions unavailable; pool1x8 available. Preparing sequential pool submissions with independent outputs and fresh capacity checks per variant.

Submitted F4: `mose-ffn-f4-r1-p1-1x8-pool-1007-0257-f50a-58255159`, pool 1x8, priority1; script `/mnt/shared-storage-gpfs2/intern-pretrain-shared02/shijiayang-p/archspace-MoSE/reproduce/olmo-core-backend/scripts/train/FFN_Rank_Reallocation/run_f4.sh`.

Submitted F5: `mose-ffn-f5-r1-p1-1x8-pool-1007-0257-2198-70783152`, pool 1x8, priority1; script `/mnt/shared-storage-gpfs2/intern-pretrain-shared02/shijiayang-p/archspace-MoSE/reproduce/olmo-core-backend/scripts/train/FFN_Rank_Reallocation/run_f5.sh`.

## 2026-10-07 03:04 HKT — F4/F5 startup verified

Both pool1x8 priority1 controls Running/active1. Mesh1x8, freshinit, initialstep0 checkpoint successfully saved, dry-run complete, finite CE/gradient atstep50. No observed training Traceback/OOM. The initial02:58 Tracebacks came from the rjob logs API returning None before pod logs were available, not from training; status was Starting/Running and no duplicate submission occurred. Exact W&B segments and proofs in jobs.json; cohort manifest now27configs. Existing routine monitor reads updated registry dynamically; no reload or GPU interruption.

## 2026-10-07 05:26 HKT — F4qk/F5qk completed

Both source jobs freshly verified Succeeded / active0 / succeeded2. Final14306 save completed and Training complete proved in completion_f4qk.log / completion_f5qk.log. Full30B W&B exports contain every step. Tail1000: F4qk2.485783 / F5qk2.479613; neither beats originalC6(2.474763). See [completed comparison](../UG_Partial_Sharing/F4_F5_QK_RESULTS.md). Upload confirmation recorded separately in synchronization.json; no repeated submission.

## 2026-10-07 08:26 HKT — F1qk/F2qk completed

Both source attempts freshly Succeeded / active0 / succeeded1, final14306 saves and Training complete proved in completion_f1qk.log / completion_f2qk.log. Exact full14306 histories exported and checked. Tail1000 F1qk2.483380551 / F2qk2.494263113. Compared withIdentity counterparts F1qk worsens+0.001238471, F2qk improves−0.004322612; neither beatsC6. [Full18-config results](../UG_Partial_Sharing/F1_F2_QK_RESULTS.md). Upload states/proofs remain separately in synchronization.json; no restart or duplicate source task.

## 2026-10-07 16:36 HKT — F4/F5 and all FFN CE statistics complete

F4/F5 Succeeded / active0 / succeeded1 and ordered final14306 save followed by checkpoint completion and Training complete verified in completion_f4.log / completion_f5.log. Exact W&B segment Finished syncing verified in sync_f4_observation.log / sync_f5_observation.log. Local unsampled scan_history contains every step1–14306; C0/C2/C6 reference statistics match archived results within1e-12. Common exclusion12516, full14305 points, tail1000. F4 tail2.486619535 / full2.799986746; F5 tail2.484029157 / full2.797487981. All ten FFN allocations remain worse than C6. Best tested FFN-only allocation F3QK(sharedUG480/Down672) tail2.477421343; P0QK(UG512 unshared/Down512) tail2.477410089 is effectively tied. C6(sharedUG568/Down520, O496) tail2.474762967 remains the best tested model allocation; its Attention budget differs from the fixedAttention512 FFN cohort. [Verified full-history comparison](analysis/F4_F5_WITH_REFERENCES_FULL_TRAIN_CE.json) / [All ten allocations](analysis/FFN_ALL_WITH_REFERENCES_FULL_TRAIN_CE.json).
