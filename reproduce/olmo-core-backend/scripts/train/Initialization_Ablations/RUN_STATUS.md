# Initialization ablation ledger — 2026-09-13 (Hong Kong)

Plan: rank512/no-sharing, Q/K/V/O/Gate/Up SiLU and Down RMSNorm. Step0 A/B/C
(independent/rawSVD/scale-matchedSVD), then full30B paired A/C. Default2×8
for full training,1×8 for preparation. No old jobs or results changed.

Local verification:5 tests passed (0.242s): exact SVD reconstruction/sign,
factor RMS matching/deterministic separate RNG, real tiny-model loader and
forward/backward, complete config defaults, real activation hooks/sampled CE.
Shell syntax and git diff whitespace checks passed. GPU verification pending.

13:53:36 resource prediction reported1/1 schedulable and14 whole8-GPU
candidate nodes; this is a partial instantaneous prediction, not allocation.
Only designated gateway/conda MoSE/arch are used; development remains local.

Step0 submitted13:54:02 with name `mose-svd-init-step0-1x8-0913` and
launcher `run_step0_probe.sh`. Actual job ID:
`mose-svd-init-step0-1x8-0913-40619762`. Runtime status is still pending.
Outputs use `scripts/analysis/Initialization_Analysis/results/svd-init-r512-20260913-v1`.
The directory cannot be overwritten, and full training requires a successful
manifest-linked probe marker. Both full training jobs are not yet submitted.

13:54:30 read-only rjob status: Inqueue,1 replica,all activity/failure counters0.
Combined14-test regression suite passed (0.698s), including6 initialization
tests and existing QK configuration/analysis checks. Probe report rejects
missing conditions and unverified FSDP initialization. No model result yet.

13:57:01 confirmed first attempt Failed,active0/failed1. Factors were generated
but nonzero ranks reached samples.pt before it existed: NumpyPaddedFSLDataset
prepare() contains a barrier, which was incorrectly called only on rank0 and
matched the other ranks' later barrier. Fixed by having every rank prepare
the identical fixed samples, with only rank0 saving and a barrier afterward.
The failed v1 directory is preserved; replacement uses fresh v2 output and
training now targets v2. Probe has no W&B run/optimizer, so no offline segment
exists to synchronize. No full-training attempts have been launched.

Replacement submitted13:57:24: `mose-svd-init-step0-r2-1x8-0913-43346781`,
arch1×8, fresh v2 artifacts. No overwrite or concurrent attempt.

User now explicitly permits4×8 acceleration for current training submissions.
After probe passes, submit A and C each4×8. GlobalBS256/micro4 means2 gradient
accumulation steps instead of4; no change to30B/LR/WSD/checkpoint cadence.
Do not cancel/restart the running1×8 preparation merely to change resources.

14:00:26 second attempt confirmed Failed,active0/failed1. All six forward
conditions completed and wrote CE/activation CSVs. Independent FSDP init
passed; matched artifact copy failed with64-vs512 local/full-row mismatch.
Core distribute_like(source,target) uses source for layout and target for
values; corrected call to distribute_like(weight,value). A real two-worker
CPU DTensor round-trip regression is being run before the next GPU attempt.
v2 is preserved; no W&B segment (analysis only), no training started.
Launchers now use a fresh v3 directory for the corrected end-to-end probe.

14:01 local Gloo2-worker DTensor regression passed: all7 tiny-model factor
pairs load into real row shards and reconstruct bitwise-identical saved factors
on both workers. No GPU used by this test; its own temporary fixture removed.
14:01:22 arch4×8 prediction reports4/4 schedulable,13 whole8GPU candidate
nodes (partial instantaneous listing, not allocated capacity). Third1×8 probe
submission requested after second attempt was confirmed terminal; no W&B sync
needed because neither analysis attempt creates a W&B run.

Third probe submitted14:01:43: `mose-svd-init-step0-r3-1x8-0913-2867578`,
arch1×8; v3 samples have appeared locally. Awaiting complete GPU smoke/report.

14:03:14 third probe completed;14:03:53 logs show SUCCEED and runner exit0.
All8 FSDP matched-factor round-trips passed,112 pairs/32 sequences/64512
activation records complete. Manifest SHA256:
`64dd7689a0b3f3e8f989507552f4288f5c91f43e1f1696d9ebc6ebc02ac62bc3`.
Local probe_complete.json status=passed and exact manifest checksum verified.
FP8 initial sampled-tokenCE: A11.941447, B11.856553, C11.854375. BF16:
A11.940384, B11.859054, C11.854536. Not evidence of final training gains.

Now submitting paired full30B A/C eacharch4×8 as explicitly authorized.
Artifacts are the verified immutable v3 output; rawSVD remains step0-only.

## Full training submissions

- A submitted14:04:16: `mose-init-independent-4x8-0913-55452062`.
  Launcher `run_independent_fp8.sh`, run `olmo3-1b-init-independent-down-rms-fp8-30b`.
- C submitted14:04:34: `mose-init-svd-matched-4x8-0913-73794777`.
  Launcher `run_svd_scale_matched_fp8.sh`, run `olmo3-1b-init-svd-scale-matched-down-rms-fp8-30b`.

Both arch4×8,30B; runtime initialization/first steps pending verification.
Checkpoint namespaces are distinct, using each run name/stage1 under existing CKPT_DIR.

14:06–14:08 startup checks: both jobs have4 running replicas and32-worker
Gloo connectivity. A rank0 logs14:05:40 mesh(dp_replicate=4,dp_shard=8) and
14:05:47 INITIALIZATION VERIFIED independent,pairs112,matching v3 manifest.
Both have advanced to evaluation-dataset preparation; first training step
still pending at the latest log observation. No retry/cancellation/sync.

C actual training initialization verified14:05:53: mode=svd_scale_matched,
pairs112,exact v3 manifest.14:07:56 reports training14306steps; step0 checkpoint
saved successfully14:08:03;14:08:04 forward/backward dry-run started.
Live W&B segment (do NOT sync while active):
`/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20260913_140802-olmo3-1b-init-svd-scale-matched-down-rms-fp8-30b`.

A step0 checkpoint saved14:07:57; forward/backward dry-run completed14:08:06,
then epoch1 started using existing global data indices/seed34521.
Live segment (do NOT sync while active):
`/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20260913_140757-olmo3-1b-init-independent-down-rms-fp8-30b`.

C confirmed optimizer-step progress through17/14306 at14:08:51; consecutive
steps12–17 advance ~0.75s each. Initial dry-run therefore passed. Full run and
final comparison remain in progress, not complete.

A confirmed optimizer-step progress through48/14306 at14:09:08, ~0.75s/step.
Both32-GPU training jobs now advance normally; no restart/sync required.

14:09:43 A progressed94/14306;14:11:21 C progressed216/14306.
Scheduler checks14:11:46(A) and14:13:09(C) each show Running,active4,failed0.
No terminal sync/restart. Added CPU-only export_training.py and strict offline
compare_training.py for eventual paired full-training reports. Three local
unittests passed (including subcases for unfinished/config/seed changes,
missing validation sources/points, nonfinite train CE, duplicate steps/token
mismatches, SVG validity, and non-overwrite). No training files or live run
configuration changed; full30B completion/sync/final analysis remain pending.

14:13:50 A logs415/14306;14:14:14 C logs443/14306, both advancing. Locally
verified factor-RMS max A/C discrepancy9.4541e-8 and summarized the step0
nonzero spectra/activation scales in
`scripts/analysis/Initialization_Analysis/STEP0_INTERPRETATION_20260913.md`.
This is diagnostic only, not final validation evidence or a causal proof.

First evaluation verified from rank0 logs: A14:15:17/C14:15:20,step500,
1,048,576,000tokens,144batches,~20.5/20.4s. Both returned to training;
latest14:16:41 A610/14306 and14:16:54 C625/14306. Rounded console CE
equal-source means across11sources A3.9235/C3.9242 (essentially tied, not
full-precision W&B or statistical evidence). Per-source figures saved in
Initialization_Analysis/EARLY_VALIDATION_20260913.md. Still warmup; no changes,
no uploads while live; first periodic checkpoint1788 remains pending.

14:19:58(A)/14:20:54(C) scheduler rechecks: both Running,active4,failed0.
Step1000 eval completed14:21:59(A)/14:21:58(C), 2.097152B tokens; rounded
11-source macroCE A3.4962/C3.4894. Stored all source values alongside step500
in EARLY_VALIDATION_20260913.md, not a final/full-precision result. Both return
to optimizer steps: A1044 at14:22:32, C1060 at14:22:43. No sync/restart/change.
Read-only local checks of logged checkpoint stage1 path and its gpfs mount
alias found neither visible here; only save-log evidence exists so far,
not local checkpoint content/integrity verification. Gateway remains query-only.

## First periodic checkpoint verified (12.5%)

Repeated read-only live-log checks14:23–14:32 confirmed continued A/C progress.
Step1500 eval completed14:28:39(A)/14:28:34(C). Rounded11-source macroCE
A3.3368/C3.3412, with C ICE CE3.788 vsA3.702; all values preserved in
EARLY_VALIDATION_20260913.md. Early gap changes sign; no superiority claim.

- A:14:32:19 started async save to its existing stage1/step1788;
  **14:32:23 "Checkpoint for step 1788 saved successfully"**.
  Continued to1799 at14:32:30.
- C: **14:32:15 "Checkpoint for step 1788 saved successfully"**;
  continued to1827 at14:32:43.

Step1788 =3,749,707,776 tokens, approximately12.5% of30B. These are explicit
completed-save runtime records, not merely start-save events. They are still
not independent checkpoint reload/integrity tests. Both runs remain active;
no W&B sync/restart or hyperparameter changes. Full14306 steps and final
synchronization/comparison remain outstanding.

Warmup endpoint verified step2000 /4.194304B tokens. Both optimizer groups
0/1/2 logLR=.0050. Eval completion14:35:24(A)/14:35:13(C),11-source rounded
macroCE A3.2430/C3.2422 (nearly tied). All per-source values appended to the
early observation file. Continued optimizer progress A2044 at14:35:58 and
C2077 at14:36:11. No upload/restart/config change; now awaiting later stable
phase, remaining periodic saves, full30B completion and full-precision report.

14:37:10(A)/14:38:13(C) scheduler rechecks Running,active4,failed0. Continued
bounded waits/live-log checks through14:42 confirmed both advance normally.
Step2500 eval /5.24288Btokens completed14:42:04(A)/14:41:47(C), rounded
macroCE A3.1485/C3.1446. All source values retained in early observations;
still a small single-seed difference, not evidence of significant superiority.
Latest optimizer progress A2523 at14:42:21 andC2565 at14:42:36. No sync/restart.

## 25% checkpoint milestone

Repeated bounded waits and read-only progress checks14:43–14:57 verified both
original jobs remain live and advance normally. Step3000 rounded macroCE
A3.0937/C3.0862; step3500 A3.0506/C3.0435. Both points have all11sources
slightly lower in C. Full per-source rounded figures preserved in early report;
still not independent-seed/statistical evidence or full-precision final results.

- A step3576 async save started14:56:15; **saved successfully14:56:20**,
  then advanced to3595 at14:56:32.
- C step3576 async save started14:55:52; **saved successfully14:55:56**,
  then advanced to3659 at14:56:57.

3576steps =7,499,415,552tokens (~25% of30B). Both retain step0,step1788 and
step3576 according to runtime save events/config; disk reload not verified.
No restart, W&B upload, or configuration change. Next periodic save5364;
continue to14306 before final sync/export/comparison.

14:57:37(A)/14:58:31(C) scheduler checks Running,active4,failed0. Read-only
live-log checks through15:02 confirm normal progress. Step4000 eval completed
15:01:58(A)/15:01:32(C),8.388608Btokens; rounded means nearly equal again
(all per-source values retained in early observations). Latest A4022 at15:02:15,
C4066 at15:02:21. No sync, restart, or training configuration changes.

15:05:20(A)/15:05:31(C) scheduler rechecks both Running,active4,failed0.
Step4500 eval completed15:08:36(A)/15:08:05(C),9.437184Btokens. Full rounded
source figures saved in EARLY_VALIDATION_20260913.md; this is preliminary
single-seed console evidence, not final/full-precision results. Training
continues A4512 at15:08:45 andC4539 at15:08:34. No sync/restart/change.

Bounded waits and read-only log checks15:09–15:15 confirm continued progress.
Step5000 eval completed15:15:14(A)/15:14:38(C),10.485760Btokens,144batches.
All11 rounded source figures retained in early observations; C lower on10,
equal on1 at console precision, not a significance/final claim. Continued
training A5012 at15:15:23,C5044 at15:15:11. Next periodic checkpoint5364.
No restart, synchronization, or configuration changes;30B still incomplete.

## 37.5% checkpoint milestone

Repeated bounded waits and read-only live-log checks15:16–15:20 confirmed
both original jobs advance normally. Step5364 =11,249,123,328tokens.

- C started async save15:19:11; **saved successfully15:19:15**, advanced to
  5418 at15:19:54.
- A started async save15:19:47; **saved successfully15:19:52**, advanced to
  5385 at15:20:06.

These are explicit runtime completed-save records, not independent disk
reload/integrity tests. A also logs checkpoint/save_async_duration_s=-inf;
retain this auxiliary timing-metric anomaly rather than treating all metrics
as finite. It does not override the explicit successful save or continued
optimizer progress; train CE/validation evidence is audited separately.
No changes, restarts, or W&B uploads while jobs remain live. Next periodic
checkpoint7152 (~50%); full14306-step completion/final comparison pending.

Step5500 eval completed15:21:52(A)/15:21:14(C),11.534336Btokens. All11
rounded source values retained; C lower10sources but higherICE by0.017.
Still preliminary/single-seed, not final/full-precision. Continued progress
A5528 at15:22:14,C5526 at15:21:34. No sync/restart/configuration changes.

15:22:46(A)/15:22:56(C) scheduler rechecks both Running,active4,failed0.
Bounded waits/log checks through15:28 show normal advancement. Step6000
eval completed15:28:29(A)/15:27:48(C),12.582912Btokens; all11 rounded
source values retained, all slightly lowerC at this point, not a final or
significance conclusion. Continued A6018 at15:28:43,C6059 at15:28:32.
No sync/restart/config changes. Remaining full training and report pending.

Read-only log checks and bounded waits15:29–15:35 confirm live progress.
Step6500 evaluation completed15:35:09(A)/15:34:21(C),13.631488Btokens.
All11 rounded source values saved; nearly tied macroCE with mixed source
directions, retained rather than selecting only stronger C points. Continued
training A6518 at15:35:22,C6568 at15:35:12. No restart/sync/config changes.

## Full training complete; terminal synchronization

Read-only scheduler checks22:47:57(C)/22:48:08(A),2026-09-13 HK, confirm
both Succeeded,active0,succeeded4,failed0. Both reached14306/14306
(30,001,856,512tokens), with runner-shim exitCode0.

- C: step14304 saved17:17:35; final step14306 saved17:17:41;
  Training complete17:17:44.
- A: step14304 saved17:19:32; final step14306 saved17:19:38;
  Training complete17:19:40.

Final checkpoint statements are runtime save confirmation, not disk reload tests.
On the subsequent train-CE analysis request, ran conda activate MoSE and
wandb_sjy for A's exact offline-run-20260913_140757 segment (PID859114).
Verified Finished syncing for that exact directory in /tmp/wandb_sjy.log.
Then started C's exact offline-run-20260913_140802 segment (PID859654);
its completion is still pending at this ledger entry. No training restart.

## Final train-CE analysis (2026-09-13)

C synchronization subsequently verified Finished syncing for its exact
offline-run-20260913_140802 segment. Both A/C terminal segments are synchronized.
No restart, resubmission, checkpoint modification, or evaluator patch was made.

Read-only W&B export completed for both finished runs: 14307 rows and207 metrics
per run, final step14306. The original GPU-owned results directory was not
writable locally; no permission changes were attempted. New ignored snapshot:
scripts/analysis/Initialization_Analysis/snapshots/training-final-20260913-2254/.

compare_training.py completed successfully: configuration differences only
factor_init and naming/output fields; all14306 paired train steps finite and
present; token/endpoint/duplicate-step and validation completeness checks passed.
Last1000 train CE A=2.495848797798157,C=2.494133560419083,
A-C=0.001715237379074; one paired seed only. Last common validation at14000
has11-source equal macro A=2.755386536771601,C=2.7521203539588233.
The earlier discovered evaluator mask-alignment caveat remains; no corrected
reevaluation has been run. See analysis/Initialization_Analysis/TRAIN_CE_20260913.md
for interpretation, windows, caveats, and snapshot artifact listing.
