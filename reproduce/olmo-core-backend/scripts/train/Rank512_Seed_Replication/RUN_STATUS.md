# C0/C6 paired initialization-seed replication

Prepared four configs; no GPU submission yet. Initialization seeds12537/12538, data seed34521. Original seed12536 remains in the completed C0/C6 histories. Full30B,797,050,880 parameters, native SwiGLU, Down RMSNorm; all other model and training settings preserved. Low priority1; submit to immediately available arch16/32 or an available whole8-GPU node as authorized.

Submitted C0S2: `mose-r512-c0s2-r1-p1-1x8-pool-1006-2106-ea9c-f2516`, pool 1x8, priority1; script `/mnt/shared-storage-gpfs2/intern-pretrain-shared02/shijiayang-p/archspace-MoSE/reproduce/olmo-core-backend/scripts/train/Rank512_Seed_Replication/run_c0s2.sh`.

Submitted C6S2: `mose-r512-c6s2-r1-p1-1x8-pool-1006-2107-3d78-52e71`, pool 1x8, priority1; script `/mnt/shared-storage-gpfs2/intern-pretrain-shared02/shijiayang-p/archspace-MoSE/reproduce/olmo-core-backend/scripts/train/Rank512_Seed_Replication/run_c6s2.sh`.

Submitted C0S3: `mose-r512-c0s3-r1-p1-1x8-pool-1006-2107-2f04-80461`, pool 1x8, priority1; script `/mnt/shared-storage-gpfs2/intern-pretrain-shared02/shijiayang-p/archspace-MoSE/reproduce/olmo-core-backend/scripts/train/Rank512_Seed_Replication/run_c0s3.sh`.

Submitted C6S3: `mose-r512-c6s3-r1-p1-1x8-pool-1006-2108-31a7-fc86e`, pool 1x8, priority1; script `/mnt/shared-storage-gpfs2/intern-pretrain-shared02/shijiayang-p/archspace-MoSE/reproduce/olmo-core-backend/scripts/train/Rank512_Seed_Replication/run_c6s3.sh`.

## 21:08 HKT — submitted paired replications

Four submissions were confirmed with unique actual scheduler IDs after positive pool1x8 predictions. arch2x8/4x8 was not schedulable. All use pool1x8, priority1, original30B recipe and797,050,880 params. Unified monitor was restarted locally with its next poll retained at21:24; GPU jobs were untouched. Submitted does not imply allocated/training; first startup verification is pending the next poll.

## 21:24 HKT — startup audit

- {"variant": "c0s2", "job_id": "mose-r512-c0s2-r1-p1-1x8-pool-1006-2106-ea9c-f2516", "exit_code": 0, "mesh": ["(dp_replicate=1, dp_shard=8)"], "segments": ["/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_210958-olmo3-1b-r512matched-c0-seed12537-data34521-fp8-30b"], "step0_saved": false, "nonfinite_ce": false, "oom": false, "traceback": false, "proof": "/mnt/shared-storage-user/intern-pretrain-shared02/shijiayang-p/archspace-MoSE/reproduce/olmo-core-backend/scripts/train/Rank512_Seed_Replication/startup_c0s2.log", "observed_at": "2026-10-06T21:25:41.344068+08:00"}
- {"variant": "c6s2", "job_id": "mose-r512-c6s2-r1-p1-1x8-pool-1006-2107-3d78-52e71", "exit_code": 0, "mesh": ["(dp_replicate=1, dp_shard=8)"], "segments": ["/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_211039-olmo3-1b-r512matched-c6-seed12537-data34521-fp8-30b"], "step0_saved": false, "nonfinite_ce": false, "oom": false, "traceback": false, "proof": "/mnt/shared-storage-user/intern-pretrain-shared02/shijiayang-p/archspace-MoSE/reproduce/olmo-core-backend/scripts/train/Rank512_Seed_Replication/startup_c6s2.log", "observed_at": "2026-10-06T21:25:42.127698+08:00"}
- {"variant": "c0s3", "job_id": "mose-r512-c0s3-r1-p1-1x8-pool-1006-2107-2f04-80461", "exit_code": 0, "mesh": ["(dp_replicate=1, dp_shard=8)"], "segments": ["/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_211054-olmo3-1b-r512matched-c0-seed12538-data34521-fp8-30b"], "step0_saved": false, "nonfinite_ce": false, "oom": false, "traceback": false, "proof": "/mnt/shared-storage-user/intern-pretrain-shared02/shijiayang-p/archspace-MoSE/reproduce/olmo-core-backend/scripts/train/Rank512_Seed_Replication/startup_c0s3.log", "observed_at": "2026-10-06T21:25:49.291665+08:00"}
- {"variant": "c6s3", "job_id": "mose-r512-c6s3-r1-p1-1x8-pool-1006-2108-31a7-fc86e", "exit_code": 0, "mesh": ["(dp_replicate=1, dp_shard=8)"], "segments": ["/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261006_211118-olmo3-1b-r512matched-c6-seed12538-data34521-fp8-30b"], "step0_saved": false, "nonfinite_ce": false, "oom": false, "traceback": false, "proof": "/mnt/shared-storage-user/intern-pretrain-shared02/shijiayang-p/archspace-MoSE/reproduce/olmo-core-backend/scripts/train/Rank512_Seed_Replication/startup_c6s3.log", "observed_at": "2026-10-06T21:25:49.445374+08:00"}

21:27 startup audit correction: async checkpoint success is logged as "Checkpoint for step 0 saved successfully". All new variants in this registry have that explicit success line, fresh initialization, (dp_replicate=1,dp_shard=8), finite CE, and no OOM/Traceback. Earlier false flags were parser mismatches; actual checkpoint writes succeeded.

Submitted C2S2: `mose-r512-c2s2-r1-p1-1x8-pool-1006-2338-833a-1f47a`, pool 1x8, priority1; script `/mnt/shared-storage-gpfs2/intern-pretrain-shared02/shijiayang-p/archspace-MoSE/reproduce/olmo-core-backend/scripts/train/Rank512_Seed_Replication/run_c2s2.sh`.

Submitted C2S3: `mose-r512-c2s3-r1-p1-1x8-pool-1006-2338-d226-b5503`, pool 1x8, priority1; script `/mnt/shared-storage-gpfs2/intern-pretrain-shared02/shijiayang-p/archspace-MoSE/reproduce/olmo-core-backend/scripts/train/Rank512_Seed_Replication/run_c2s3.sh`.

## 2026-10-07 09:26 HKT — C0 seeds completed

C0 init12537/12538 source attempts freshly Succeeded / active0 / succeeded1, final14306 saves and Training complete verified in completion_c0s2.log / completion_c0s3.log. Fixed data34521 and original797050880 budget preserved. Uploads serialized and tracked in synchronization.json. CorrespondingC6 seeds are still active at scheduled poll; no paired-seed quality conclusion until their complete histories are available.

## 2026-10-07 10:32 HKT — C0/C6 paired-seed confirmation

All4 source attempts Succeeded/active0/final14306 saved and exact W&B uploads Finished syncing. init12537/12538, fixeddata34521, samepool1x8,797050880params; complete Train CE and corrected final LM analysis jobs both Succeeded/active0/succeeded1. Mean paired C6-minus-C0 last1000Train CE -0.004087218; final11-source corrected macro CE -0.006055193; both init seeds improve on bothmetrics. Four final JSON/SHA256 verified with identical per-source counts; C2 replications stillRunning, noC6-versus-C2 repeat conclusion. Full results: C0_C6_RESULTS.md.

## 2026-10-07 11:57 HKT — C2 seed12537 complete

`mose-r512-c2s2-r1-p1-1x8-pool-1006-2338-833a-1f47a` verified Succeeded/active0/succeeded1, final14306 checkpoint saved before Training complete; proof completion_c2s2.log. Exact W&B synchronization state: finished. C2 seed12538 remainsRunning; complete two-seed C0/C2/C6 Train CE and final-checkpoint validation await its terminal proof.

## 2026-10-07 12:36 HKT — all6 seed replications and final LM complete

All C0/C2/C6 init12537/12538 source runs verified Succeeded/active0/final14306 saved and exact W&B Finished syncing. Combined read-only pool1x8 priority1 analysis `mose-seed-confirm-r1-p1-1x8-pool-1007-1228-6-a9b14` Succeeded/active0/succeeded1; complete6 Train CE and both C2 final LM outputs SHA256 verified. Same data34521,797050880params,trainingpool1x8; no train omissions/anomalies; tail1000 all1000points. Two-seed means: C0/C2/C6 Train CE 2.479431683/2.475767208/2.475344465; corrected final11-source macro CE 2.740639253/2.733279575/2.734584061. Both seeds favor C6 train and C2 validation; C6-minus-C2 mean Train -0.000422743, validation+0.001304486. C0/C6 validation from prior completed allocation; C2 from currentallocation, identical per-source counts. No significance claim. Full report C0_C2_C6_RESULTS.md; previous four-run report/JSON/figures preserved; new6-point figure visually checked.
