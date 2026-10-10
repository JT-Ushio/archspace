# Attention rank budget reallocation

A1/A2 prepared locally; validation and available allocation required before submission. No training job submitted yet.

Actual module/meta parameter budgets and identical training recipe validated: 1passed166.67s. Global batch supports8/16/32GPU. All launchers bash-n passed. No job submitted yet.

Submitted A1: `mose-attn-a1-r1-p1-1x8-pool-1007-0128-5292-1-78ab4`, pool 1x8, priority1; script `/mnt/shared-storage-gpfs2/intern-pretrain-shared02/shijiayang-p/archspace-MoSE/reproduce/olmo-core-backend/scripts/train/Attention_Rank_Reallocation/run_a1.sh`.

Submitted A2: `mose-attn-a2-r1-p1-1x8-pool-1007-0128-b35f-2-1cf16`, pool 1x8, priority1; script `/mnt/shared-storage-gpfs2/intern-pretrain-shared02/shijiayang-p/archspace-MoSE/reproduce/olmo-core-backend/scripts/train/Attention_Rank_Reallocation/run_a2.sh`.

## 2026-10-07 01:31 HKT — allocated, initialization underway

Both A1/A2 submitted once to available pool1x8, priority1. Each Running/active1 with dp_replicate1/dp_shard8. No observed startup errors; step0, exact W&B segments and finite training CE still pending. Global monitor reloaded locally to include this family; next poll01:56 HKT. No training job interrupted.

## 2026-10-07 02:14 HKT — startup fully verified

A1/A2 Running/active1, finite CE at steps810/820, mesh1x8, fresh initialization, initialstep0 checkpoint successfully saved asynchronously, dry-run complete. Exact W&B segments and proofs stored in jobs.json and startup_a*_verified.log; added to cohort manifest. Routine monitor next02:26, then every30min.

## 2026-10-07 16:28 HKT — A1/A2 complete Train CE verified

Both source attempts Succeeded / active0 / succeeded1, with ordered final step14306 save, successful checkpoint completion and Training complete verified in completion_a1.log / completion_a2.log. Exact W&B segments Finished syncing; invocation and observation proofs retained. Local W&B API scan_history read every Train CE row at steps1–14306 for A1/A2 and C0/C2/C6, without sampling. Archived reference statistics match within1e-12. Common exclusion step12516; 14305 full-window points and1000 tail points. Tail CE A1=2.498468154, A2=2.498579926, C6=2.474762967; deltas +0.023705187 / +0.023816959. Full means A1=2.806920823, A2=2.808385675. Neither allocation improves the best tested low-rank setting. No new final validation claim. [Raw histories, checksums and comparison](analysis/A1_A2_WITH_REFERENCES_FULL_TRAIN_CE.json).
