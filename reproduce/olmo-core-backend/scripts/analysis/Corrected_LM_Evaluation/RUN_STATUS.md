# Standalone final-checkpoint mask reevaluation

Source inspection confirmed get_labels masks tokens before shifting targets, while LMEvaluator selects the original input mask. Isolated evaluation compares both masks on one forward/data sequence; training/core files unchanged. Both exact final checkpoint paths gated by current completion proofs. Alignment tests5passed59.99s; syntax checks passed.

Submitted COMPLETED-PAIR-MASK-CHECK: `mose-eval-mask-pair-r1-p1-1x8-pool-1007-0342-54a01`, pool 1x8, priority1; script `/mnt/shared-storage-gpfs2/intern-pretrain-shared02/shijiayang-p/archspace-MoSE/reproduce/olmo-core-backend/scripts/analysis/Corrected_LM_Evaluation/run_completed_pair.sh`.

Observation 2026-10-07T03:46:34.419872+08:00: Running, {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; 8-rank mesh and exact F3qk step14306 load observed. Evaluation outputs/completion not yet verified. Proof: startup_observation.log.

03:49 HKT: Succeeded active0/succeeded1 verified; both exact final14306 checkpoints loaded and both11-source results recovered with individual SHA256. Corrected macroP0qk=2.736911958,F3qk=2.737175356; same-forward legacy=2.730046890/2.730310960. See RESULTS.md.

Submitted REFERENCE-PATH-RESOLVE: `mose-ref-paths-r1-p1-1x8-pool-1007-0354-07cc-88d55`, pool 1x8, priority1; script `/mnt/shared-storage-gpfs2/intern-pretrain-shared02/shijiayang-p/archspace-MoSE/reproduce/olmo-core-backend/scripts/analysis/Corrected_LM_Evaluation/run_resolve_references.sh`.

03:56 HKT: reference-path-resolve Succeeded active0/succeeded1; complete C0/C2/C6 W&B histories, actual save_folder and final config.json model/seed matched. REFERENCE_CHECKPOINTS.json recovered with SHA256. Fresh rjob queries of archived source IDs returned no state; resolution uses recorded completion and actual metadata, not fresh source terminal claims.

Submitted COMPLETED-REFERENCE-TRIO: `mose-eval-ref-trio-r1-p1-1x8-pool-1007-0359-618bb`, pool 1x8, priority1; script `/mnt/shared-storage-gpfs2/intern-pretrain-shared02/shijiayang-p/archspace-MoSE/reproduce/olmo-core-backend/scripts/analysis/Corrected_LM_Evaluation/run_reference_trio.sh`.

04:07 HKT: reference trio Succeeded active0/succeeded1; final14306 metadata/model/seed matched and all three11-source results individually SHA256 verified. Corrected macroC0=2.738865972,C2=2.732681404,C6=2.734855153. C2 remains best validation, C6 best TrainCE. Five-case per-source token counts identical. See RESULTS.md and FIVE_CONFIG_FINAL_CHECKPOINT_LM.json.

Submitted FFN-QK-FINAL-PAIR: `mose-eval-ffn-qk-r1-p1-1x8-arch-1007-0528-63-fc7d7`, arch 1x8, priority1; script `/mnt/shared-storage-gpfs2/intern-pretrain-shared02/shijiayang-p/archspace-MoSE/reproduce/olmo-core-backend/scripts/analysis/Corrected_LM_Evaluation/run_ffn_qk_pair.sh`.

## 2026-10-07 05:33 HKT — F4qk/F5qk final evaluation completed

`mose-eval-ffn-qk-r1-p1-1x8-arch-1007-0528-63-fc7d7` freshly Succeeded / active0 / succeeded1. Both final14306 checkpoints loaded, recipe model/seed and797050880 params guarded. Each11-source JSON checksum verified; per-source label counts identical to previous5 cases. Corrected macro CE F4qk2.745877028 / F5qk2.738925078; originalC2 remains validation best. No training or checkpoint saving. Results and7-case summary updated,5-case snapshot preserved.

Submitted ORIGINAL-DENSE-PATH-RESOLVE: `mose-resolve-dense-r1-p1-1x8-arch-1007-0537-4cee7`, arch 1x8, priority1; script `/mnt/shared-storage-gpfs2/intern-pretrain-shared02/shijiayang-p/archspace-MoSE/reproduce/olmo-core-backend/scripts/analysis/Corrected_LM_Evaluation/run_resolve_dense.sh`.

Submitted ORIGINAL-DENSE-FINAL-LM: `mose-eval-dense-r1-p1-1x8-arch-1007-0539-5a2-5fbc0`, arch 1x8, priority1; script `/mnt/shared-storage-gpfs2/intern-pretrain-shared02/shijiayang-p/archspace-MoSE/reproduce/olmo-core-backend/scripts/analysis/Corrected_LM_Evaluation/run_dense_final.sh`.

## 2026-10-07 05:39 HKT — original Dense source resolved; evaluation submitted

Resolver `mose-resolve-dense-r1-p1-1x8-arch-1007-0537-4cee7` verified Succeeded / active0 / succeeded1, payload checksum a9ededb696ca3e7be6f3985d2966e494a41b4cddabf4eff28d4fe677c7a6bbe0. Actual W&B save_folder final14306 metadata exactly matches original Dense recipe/seed,1484916736 params. Two resumed segments cover all14306 steps; newer records override3474 overlapping steps, exactly reproducing prior Dense full/tail/final TrainCE. Original source scheduler job ID unavailable (null), no fresh terminal claim. Final8-rank corrected/legacy evaluation submitted on availablearch8 priority1, same32768 microbatch/data partition as low-rank reevaluations. No training or checkpoint saving.

## 2026-10-07 05:57 HKT — original Dense final LM completed

`mose-eval-dense-r1-p1-1x8-arch-1007-0539-5a2-5fbc0` verified Succeeded / active0 / succeeded1. Actual final14306 checkpoint loaded; saved model/seed and1484916736 params checked. SHA2568015dc2a756b57b5f417d8dc0c4d0b487c3c77c5565863c5663d3136cd093614. CorrectedmacroCE2.627212026 / legacysameforward2.620586710. Every11-source valid/legacy label count identical to low-rank evaluations. Uniform8ranks/micro32768. Source scheduler ID unknown explicitly; no original terminal-state claim. Dense comparison report and projectoverview updated.

Submitted C0-C6-SEED-FINAL-LM: `mose-eval-seed-c0c6-r1-p1-1x8-pool-1007-0958-9a0c4`, pool 1x8, priority1; script `/mnt/shared-storage-gpfs2/intern-pretrain-shared02/shijiayang-p/archspace-MoSE/reproduce/olmo-core-backend/scripts/analysis/Corrected_LM_Evaluation/run_c0_c6_seed_final.sh`.

Submitted ALL-SEED-TRAIN-AND-C2-FINAL-LM: `mose-seed-confirm-r1-p1-1x8-pool-1007-1228-6-a9b14`, pool 1x8, priority1; script `/mnt/shared-storage-gpfs2/intern-pretrain-shared02/shijiayang-p/archspace-MoSE/reproduce/olmo-core-backend/scripts/analysis/Corrected_LM_Evaluation/run_all_seed_train_and_c2_final.sh`.
