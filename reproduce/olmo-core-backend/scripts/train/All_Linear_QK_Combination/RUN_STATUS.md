# Next-round ledger — 2026-09-12 (Asia/Hong_Kong)

## Training

- Submitted via designated SSH gateway after `conda activate MoSE`.
- `gpu_run arch 2 8 mose-all-linear-qk-silu-2x8-0912 .../run_qk_silu_fp8.sh`.
- Actual job: `mose-all-linear-qk-silu-2x8-0912-34757738`, created01:07:15.
- 01:33:29 SSH rjob get confirmed Inqueue; 2 replicas,0 active,0 failed,
  0 succeeded,0 stopped. Rechecked02:28:03: still Inqueue. Waiting for allocation;
  not a stopped/failed attempt.
- Local validation: 8 tests passed (1.667s), launcher `bash -n` passed.
- New run/checkpoint namespace: `olmo3-1b-all-linear-qk-silu-fp8-30b/stage1`.

## Analysis

- Submitted01:15:29 via MoSE/gpu_run arch2x8.
- Actual job: `mose-qk-checkpoint-analysis-2x8-0912-29594700`.
- Launcher: `scripts/train/QK_Combination_Analysis/run_qk_checkpoint_analysis.sh`.
- Read-only baseline/Q-only/K-only at exact3576/7152/10728/14306,128 fixed
  held-out sequences split across16 GPUs; paired BF16/FP8 inference.
- Samples, token CE, post-norm/post-RoPE attention and automatic CSV/SVG/report
  outputs use a new directory under
  `scripts/analysis/QK_Combination_Analysis/results/mose-qk-checkpoint-analysis-2x8-0912-29594700`.
- 01:33:29 SSH rjob get confirmed Inqueue; 2 replicas,0 active,0 failed,
  0 succeeded,0 stopped. Rechecked02:28:03: still Inqueue. No runtime outputs yet.
- Six analysis tests cover real small-model forward/hook/sample-logits matching,
  causal and sliding-window masks, scored-label masks, real-config auditing,
  paired aggregation/plots and rejection of duplicate assignments.

User explicitly authorized SSH queries in the latest message. Reconnected and
activated MoSE; queried both exact job IDs read-only. Neither was resubmitted.
Do not infer a stopped state from queueing or an observation timeout.
Both `rjob events` queries (rechecked02:16) returned only acceptance into the external scheduler
queue (event timestamps09-11 17:07:15/17:15:29, corresponding to submission times
in UTC). No launch-failure event or queue-position estimate was returned.

Final local test sweep:14 tests passed in1.156s (analysis, QK combination,
existing single-attention ablations and all-linear invariants); both launchers
pass bash syntax checks, and git diff --check passes. GPU runtime verification
and actual completion remain pending. No unrelated jobs or old outputs changed.

02:03 update: while queued, exported five completed L/Q/K/V/O W&B histories
read-only into QK_Combination_Analysis/results/controls-wandb-20260912-0159.
All five exports completed with14307 rows each; control_trends.py produced
matched-step CSVs/SVGs/report and passed3 targeted tests. See
QK_Combination_Analysis/CONTROL_RESULTS_20260912.md: Q/K have lower terminal
train CE but slightly worse last-validation macro CE than L. Retain the QK
combination test and checkpoint analysis; do not presume validation gains or
launch further QK+V/O combinations without results. No new GPU jobs submitted.

02:40:42 resource query: CLI help verified `rjob submit --predict-only=true`
only predicts resources. Queried arch/llmarchitecture_gpu/private-machine=group,
2 replicas ×8 GPUs,32CPU/262144MiB per replica, priority1, normal, host networking,
fuse1/RDMA8/mlnx_rdma1 and the existing image. It returned2/2 schedulable,
listing three candidate whole8GPU nodes (H2000227/2271/2257). These are partial
instantaneous candidates, not full inventory, reserved resources or a dispatch
guarantee. Prediction omitted data mounts and did not create a job (empty
created-rjob-name field). Do not assert that queueing proves zero free GPUs;
the existing tasks still await actual external-scheduler dispatch.

03:03:14 SSH status recheck: both exact jobs remain Inqueue, each with2 replicas
and zero active/succeeded/failed/created_failed/pending/stopped replicas. No
resubmission or W&B synchronization was triggered. The03:00:49 prediction
again reported2/2 schedulable, now displaying two whole8GPU candidates
(H2000227/2257); this remains a partial prediction, not an actual allocation.

03:12:29: both jobs still Inqueue, zero active/failed replicas. Events rechecked
03:08 still contain only external-queue acceptance. Combined local suite now
passes17 tests in1.122s (including control_trends); the first invocation omitted
the project src directory from PYTHONPATH, then the corrected invocation passed.
GPU startup and runtime verification remain pending.

04:00:00: repeated authorized SSH checks confirm both original jobs remain
Inqueue with zero active/failed replicas. Events rechecked03:32 still show
only external-queue acceptance. Resource prediction03:45 reports2/2 schedulable
with H2002257/0227 as partial whole8GPU candidates, not allocations. No new job,
restart, cancellation, W&B upload, or experiment-configuration change occurred.

04:59:27: both original jobs are still Inqueue, with all replica counters zero.
Both event streams rechecked04:51 contain only external-queue acceptance, with
no new error or startup event. No duplicate submissions, restarts or uploads.
This is a verified wait for the existing scheduler jobs; GPU runtime and final
results are not yet available.

06:01:27: both original job IDs remain Inqueue with all replica counters zero.
Events rechecked05:54 still show only queue acceptance. At05:04 the status API
briefly returned429 (scheduler list capacity full); retry05:05 succeeded and
confirmed queueing. Polling was reduced to approximately once every2 minutes.
The05:01 predict-only query reported2/2 schedulable and six partial whole8GPU
candidates (2257/2176/0005/2197/1143/0227), not actual allocation; no job created.
No restart, cancellation, additional submission, W&B upload or hyperparameter
change. GPU startup, runtime verification and final results remain pending.

07:00:41: both original jobs remain Inqueue, each with all replica counters
zero. Events rechecked06:55 still contain only external-queue acceptance.
The06:57 predict-only query reported2/2 schedulable, listing seven partial
whole8GPU candidates (2257/2176/2197/2170/0005/0227/2288); these are not
reservations or proof of dispatch, and the prediction created no job. No new
submission, restart, cancellation, W&B upload or experiment change occurred.
Continue monitoring the original IDs; no new runtime results are available.

07:59:56: both original jobs remain Inqueue with all replica counters zero.
Events rechecked07:55 still show only external-queue acceptance. At07:57 the
predict-only query reported2/2 schedulable and17 partial whole8GPU candidate
nodes; it created no job and does not prove actual allocation or explain the
external queue delay. Neither original job has started. No new submission,
restart, cancellation, W&B upload or hyperparameter change occurred. Continue
the verified wait; training completion and checkpoint-analysis outputs remain
unverified and pending.

08:56:56: both original jobs remain Inqueue with all replica counters zero.
Events rechecked08:54 still show only external-queue acceptance. The08:56
predict-only query reported2/2 schedulable and18 partial whole8GPU candidate
nodes; no job was created. This instantaneous resource prediction is not an
allocation, queue-position report or explanation of the dispatch delay.
No new submission, restart, cancellation, W&B upload or experiment change.
Continue monitoring the original IDs; GPU runtime and final results remain
pending. Status polling during this hour was approximately once per minute.

09:57: both original jobs remain Inqueue with all replica counters zero.
Events rechecked09:54 still show only external-queue acceptance. The09:57
predict-only query reported2/2 schedulable and18 partial whole8GPU candidate
nodes; the created job name was empty, so no job was submitted. The prediction
does not establish allocation, queue position or a cause for the queue delay.
No restart, cancellation, W&B upload or experiment change occurred. Continue
the verified wait on the original IDs; training and analysis results remain
pending.

10:57: both original jobs remain Inqueue with all replica counters zero.
Events rechecked10:55 still show only external-queue acceptance. The10:57
predict-only query reported2/2 schedulable and18 partial whole8GPU candidate
nodes; it created no job. Candidate hosts changed, but this is not evidence
of an allocation or an explanation for the queue delay. No restart,
cancellation, W&B upload, new submission or experiment change occurred.
Continue monitoring the original IDs; no runtime results are available yet.

11:56: both original jobs remain Inqueue with all replica counters zero.
Events rechecked11:56 still show only external-queue acceptance. The11:56
predict-only query reported2/2 schedulable and14 partial whole8GPU candidate
nodes, compared with18 in the preceding snapshot; it created no job. These
partial instantaneous candidates are not a full inventory, reservation or
dispatch guarantee, and do not establish the reason for the queue delay.
No restart, cancellation, W&B upload, new submission or experiment change.
Continue monitoring the original IDs; GPU execution and results remain pending.

12:55: both original jobs remain Inqueue with all replica counters zero.
Events rechecked12:55 still show only external-queue acceptance. The12:55
predict-only query reported2/2 schedulable and17 partial whole8GPU candidate
nodes; the created job name was empty and no job was submitted. This is an
instantaneous partial resource prediction, not a reservation, queue position,
dispatch guarantee or diagnosis of the external scheduler delay. No restart,
cancellation, W&B upload or experiment change occurred. Continue the verified
wait on the original IDs; training and checkpoint-analysis results are pending.

13:57: the latest job query (13:54:53) still shows both original jobs Inqueue
with all replica counters zero. Events rechecked immediately afterwards still
contain only external-queue acceptance. The 13:57 predict-only query reported
2/2 schedulable and 16 partial whole-8-GPU candidate nodes; its created job name
was empty. This did not submit or reserve resources and does not explain the
external scheduler delay. No restart, cancellation, W&B upload or experiment
change. Continue the verified wait on the original IDs.

14:55: both original jobs remain Inqueue with all replica counters zero
(status query 14:54:58). Events rechecked 14:55 still contain only external-
queue acceptance. The 14:55 predict-only query reported 2/2 schedulable and
13 partial whole-8-GPU candidate nodes; the created job name was empty.
These instantaneous candidates are not a reservation, complete inventory,
dispatch guarantee or explanation for the external queue delay. No restart,
cancellation, W&B upload, new submission or experimental configuration change.
Continue the verified wait; training and checkpoint-analysis results remain
pending actual GPU execution.

15:55: both original jobs remain Inqueue with all replica counters zero
(status query 15:54:44). Events still contain only external-queue acceptance.
The 15:55:07 predict-only check changed to 0/2 schedulable, 2/2 unschedulable;
the displayed leading nodes have only 4, 3, 2, 2, 2 and 1 free GPUs. The
created job name was empty. This supports insufficient matching resources
at this instant, not a diagnosis of the entire preceding queue delay or a
complete cluster inventory. Keep both original 2x8 jobs queued; no restart,
cancellation, W&B upload, additional submission or experiment change occurred.

16:20:31: scheduling has begun on the original IDs. Training is Starting:
cfcd2 is RUNNING on gpu-l-lg-cmc-h-h200-0215 (10.102.244.31), c4ca4 is
STARTING on gpu-lg-cmc-h-h200-0227 (10.102.204.17). The analysis job still
reports Inqueue but now lists cfcd2 and c4ca4 as STARTING (no host yet).
At 16:20:57 the public-endpoint training logs report both replicas RUNNING;
mount preparation and torchrun launch have completed and distributed
initialization is underway. No optimizer step is verified yet. Analysis log
retrieval at 16:21:02 fails inside the rjob client because log data is None
for a STARTING replica; this is not evidence that the analysis job failed.
Continue querying the same IDs; no new submission, restart or sync occurred.

16:25:38: training has reached step72/14306 on the original 2x8 job.
Runtime logs verify FP8 conversion (lm_head.w_out excluded), Muon, world_size16,
and the expected warmup: LR=0.000075 at step30 with 62,914,560 total tokens.
Step1 completed at 16:23:55. The exact active offline W&B segment is
`/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20260912_162314-olmo3-1b-all-linear-qk-silu-fp8-30b`.
Do not sync/restart this still-writing attempt.

Analysis completed at 16:24:09 (ALL ANALYSIS WORKERS COMPLETE); both replicas
report SUCCEED and runner-shim ExitCode0. Local verification confirms complete
manifest, 16 disjoint completion markers, 128 sequences, 3,072 sequence-level
condition scores, matching source hashes and 10 parseable SVGs. There are 15,070
label-valid sampled tokens per condition. See the analysis directory's
`CHECKPOINT_RESULTS_20260912.md` for results and limitations.
The 16:24 status API timeout was an observation failure, not a job failure;
public logs confirmed real training progress and analysis success. No restart,
additional submission, cancellation or W&B sync occurred.

16:37:10: first periodic validation at optimizer step500 completed at 16:36:45
(289 evaluation batches, 38.9 seconds), and training resumed through step517.
Finite validation CE values are present in the console. This verifies the
evaluation/return-to-training path, not model superiority or final completion.
No OOM or fatal error is observed in this segment; the synchronous CUDA warning
did not prevent training from continuing. The original training ID remains live.

17:10: checkpoint milestone verified in rank0 logs. Step1788 asynchronous save
started at 17:09:25 and reported `Checkpoint for step 1788 saved successfully`
at 17:09:29; training subsequently reached step1819. The saved path is
`/mnt/shared-storage-gpfs2/intern-pretrain-shared02/taoji/ckpts/Olmo3-MoSE/olmo3-1b-all-linear-qk-silu-fp8-30b/stage1/step1788`.
This verifies the trainer's successful-save result. Direct local directory
checks at that path and the shared-storage-user/intern-pretrain-shared02 alias
both returned no such directory, so independent on-disk inspection is not
verified from this development host. Do not interpret local path visibility as
a failed GPU save. No remote filesystem exploration or additional job occurred.
The step1500 validation also completed (289 batches, 33.5 seconds) and training
resumed. Training remains incomplete; no W&B synchronization or restart yet.

17:15:47: warmup milestone verified. At step2000 the console reports LR=0.0050
for optimizer groups 0/1/2. The fourth periodic validation completed at 17:15:15
(289 batches, 35.7 seconds), with finite CE for all 11 sources. Training then
resumed through step2022, and LR remained 0.0050 at steps2010 and2020. These
runtime observations match the requested 2000-step warmup and peak LR; final
performance and completion remain pending. No configuration change or restart.

17:22:45: first restart submitted after authoritative terminal verification.
Original job `mose-all-linear-qk-silu-2x8-0912-34757738` received SIGTERM around
17:20:49; DataLoader termination and torchrun signal15 were logged. Scheduler
events show both pods deleted at 17:20:47 and Running -> Stopped at 17:21:23.
The 17:21:52 status query confirms Stopped, active0, stopped2. Events do not
identify the initiator; preemption is not established. Last monitored training
step was2214; step1788 is the latest verified successful checkpoint.

Immediately after terminal confirmation, invoked `conda activate MoSE &&
wandb_sjy` with the exact offline segment recorded above. It launched as shell
job2 / PID474554; upload completion remains unverified, do not submit it again.
Then submitted the unchanged launcher with arch2x8 and new scheduler name
`mose-all-linear-qk-silu-r2-2x8-0912`; created ID is
`mose-all-linear-qk-silu-r2-2x8-0912-64469247`. RUN_NAME, save folder, optimizer,
batch, schedule and30B budget are unchanged. Local trainer source confirms
save-folder auto-resume loads trainer and optimizer state. Actual restoration
and subsequent training must still be verified from the replacement's logs.
The completed checkpoint-analysis job was not restarted. No cancellation or
model/launcher edit occurred in this recovery.

17:23: first-attempt W&B upload completion verified in the known wrapper output
`/tmp/wandb_sjy.log`: `Finished syncing` names the exact 20260912_162314 QK
segment. The uploader reported history steps0-2220 (W&B history index, not
independently asserted optimizer step). No EOF/error is shown for this sync.
Replacement status at17:23:20 remains Inqueue but has two STARTING replicas
without assigned hosts. Restore progress remains pending; do not resubmit.

17:26:58: replacement restore verified at runtime. Rank0 (now cfcd2 on H2000227)
loaded the original save folder's step1788 checkpoint beginning17:25:57, then
completed optimizer step1789 at17:26:41 and advanced through1800. This also
provides a real checkpoint-read/continue check despite the development host's
missing path visibility. Same 14306-step endpoint. The new active offline
segment is
`/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20260912_172606-olmo3-1b-all-linear-qk-silu-fp8-30b`.
Do not synchronize this still-writing segment. The previous segment is fully
synced. Final analysis must account for replayed optimizer steps after1788;
W&B history index alone must not be treated as optimizer step or unique tokens.

17:32:53: replacement completed its step2000 validation (289 batches,35.7sec)
and resumed through2019. Replayed step2000 validation is not bit-identical to
the original attempt: e.g. console c4_en CE3.548 vs3.551, ice3.581 vs3.565.
Do not treat replay as an independent seed or average both attempts at the
same optimizer step. Final curve assembly should retain the continuing
trajectory after the step1788 restore and preserve attempt provenance; the
cause of numerical divergence has not been isolated. No fatal error observed.

17:37:24: replacement also stopped. Both pods were deleted at17:36:24;
DataLoader SIGTERM exceptions followed at17:36:26. The17:36:42 scheduler query
still had one active replica, so no sync/restart was attempted then. At17:37:24
status is Stopped,active0,stopped2. Last observed training step2119; no newer
successful checkpoint than1788 is verified. Early startup configmap/conflict
events preceded successful restore/training and do not explain the later stop.
The initiator of both stop events remains unidentified. Invoked wandb_sjy once
for the exact new 20260912_172606 segment after terminal confirmation. Its
completion is pending. Hold a third submission while asking whether these
repeated pod deletions reflect an intentional user/admin stop; do not assume
preemption or circumvent a deliberate stop. Analysis remains fully complete.

17:38: second-attempt sync (PID477948) now verified complete: the known wrapper
log reports `Finished syncing` for the exact 20260912_172606 segment. The
reported W&B history range2221-2593 illustrates append-after-restart indexing,
not optimizer progress2593. Both known segments are now synced. Replacement
remains Stopped,active0 (17:37:54 status). No third submission; waiting for
confirmation that repeated stops were not an intentional user/admin request.

## 2026-09-13: user-authorized resubmission and continued tracking

User confirmed `提交并且持续追踪`. At00:32:06 the previous r2 job is still
Stopped,active0,stopped2; both previous W&B segments have verified completed
syncs and were not uploaded again. Re-read and bash-validated the unchanged
launcher locally. After activating MoSE on the authorized gateway, submitted
`gpu_run arch 2 8 mose-all-linear-qk-silu-r3-2x8-0913` with the original absolute
launcher path. Submission at00:32:21 returned
`mose-all-linear-qk-silu-r3-2x8-0913-40549445`.
Same RUN_NAME/checkpoint folder, BS256x8K,30B,Muon0.005,WSD2000/20%,FP8 and
1788-step checkpoint interval. Actual scheduling, step1788 restore and new
offline segment remain to be verified. No model/config change, cancellation,
or restart of the already-completed analysis job.

00:34:45: r3 is Running,active2 on H200-2159 and H200-2180. Rank0 is
c4ca4/H200-2180. At00:35:24 the trainer reports `Checkpoint successfully
loaded`, resumes step1788/epoch1 and data-loader batch1788 with2097152
tokens/batch. Actual resumed training reached step1793/14306 at00:36:13.
The new offline segment is
`/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20260913_003525-olmo3-1b-all-linear-qk-silu-fp8-30b`.
No new checkpoint save or final completion claimed; continue monitoring.

01:22:23: r3 reports `Checkpoint for step 3576 saved successfully`, after
async save started at01:22:19. Path is the unchanged stage1/step3576 folder
under the checkpoint root above. Training continued through3586/14306 at
01:22:36. This is the newest verified saved checkpoint (about7.5B tokens).
Step2000/2500/3000/3500 evaluations have been followed by continued training.
User permitted4x8 for this experiment; asked whether to stop the live2x8
attempt to switch, but no cancellation confirmation received. Kept2x8 live;
no duplicate job, cancellation, or new W&B upload performed.

02:07:54: r3 reports `Checkpoint for step 5364 saved successfully`, after
async save started at02:07:51 to the same stage1/step5364 checkpoint folder.
Training continued through5385/14306 at02:08:25. This is now the newest
verified checkpoint (about11.25B tokens,37.5% of the30B budget).
Step4000/4500/5000 evaluations completed and were followed by training.
The current attempt remains2x8; no confirmation to cancel/switch allocation
has arrived, so no cancellation, duplicate submission or active-segment sync
was performed. Full training and final W&B comparison remain pending.

02:54:01: r3 reports `Checkpoint for step 7152 saved successfully`, after
async save started at02:53:57 to the unchanged stage1/step7152 directory.
Training continued through7172/14306 at02:54:29. The newest verified saved
checkpoint is now approximately15B tokens (half of the30B budget).
Step5500/6000/6500 evaluations reported completion and continued training;
the7000 evaluation reached280/289 before the next poll showed resumed7045.
No errors appeared in the inspected log tails. Current allocation remains2x8;
no cancellation, duplicate submission, or active-attempt W&B sync performed.
Full training, final checkpoint, final sync and QK comparison remain pending.

03:39:33: r3 reports `Checkpoint for step 8940 saved successfully`, after
async save started at03:39:28 to the unchanged stage1/step8940 directory.
Training continued through8946/14306 at03:39:41. This is now the newest
verified saved checkpoint (approximately18.75B tokens,62.5% of the budget).
Step7500/8000/8500 evaluations completed in36.5/34.9/36.6 seconds and were
followed by resumed training. No errors appeared in the inspected tails.
The same2x8 attempt remains running; no cancellation, duplicate submission,
or active-segment sync performed. Full30B training and final reporting pending.

04:25:37: r3 reports `Checkpoint for step 10728 saved successfully`, after
async save started at04:25:33 to the unchanged stage1/step10728 directory.
Training continued through10745/14306 at04:26:02. The newest verified saved
checkpoint is approximately22.5B tokens (75% of the30B budget).
Step9000/9500/10000 evaluations completed and resumed training; continued
training beyond10500 was also observed. No errors in the inspected log tails.
Allocation remains2x8; no cancellation, duplicate submission, or active-segment
sync performed. Full training, final checkpoint, sync and comparison pending.

05:12:32: r3 reports `Checkpoint for step 12516 saved successfully`, after
async save started at05:12:28 to the unchanged stage1/step12516 directory.
Training continued through12535/14306 at05:12:59. This is the newest verified
saved checkpoint, approximately26.25B tokens (87.5% of the30B budget).
Step11000/11500/12000/12500 LM evaluations completed in35.2/36.0/35.2/36.0
seconds respectively; the12500 downstream evaluation was followed by resumed
training before the checkpoint save. No errors in inspected log tails.
The same2x8 attempt remains active; no cancellation, duplicate submission,
or active-segment sync. Final training, checkpoint, sync and comparison pending.

05:58:43: r3 is **Succeeded**, active=0, succeeded=2, failed=0. Rank0 reported
14306/14306 at05:58:08, `Training complete` at05:58:15 and runner exitCode0
at05:58:23. Periodic step14304 saved successfully at05:58:06; final synchronous
step14306 save started05:58:09 and reported `Checkpoint saved` at05:58:13.
The final path is the unchanged stage1/step14306; total tokens=30,001,856,512.
Step14000 LM evaluation completed05:50:37 in36.6 seconds. Training stayed2x8;
no switch to4x8 or duplicate writer was launched.

After terminal/zero-active verification, invoked `conda activate MoSE && wandb_sjy`
for the exact r3 segment `offline-run-20260913_003525-olmo3-1b-all-linear-qk-silu-fp8-30b`
under the previously recorded W&B root. Background PID635495; first tail confirms
this segment is being synchronized, not yet completion. Final sync/export and
matched QK comparison remain pending; the existing checkpoint-analysis job is
also verified Succeeded with2 successful replicas and is not resubmitted.

Final handoff, 2026-09-13: exact r3 segment `Finished syncing` verified in the
authorized wrapper log; original/r2 segments were already synchronized. W&B
read-only final export reports finished,14307 rows/208 keys, step0–14306.
All non-null token counts match step×2,097,152; every post-restore step1789–14306
has a timestamp from successful r3. API output has no duplicate steps, so no
deduplication/renumbering/averaging was applied. Upload transport indices
2594–15112 must not be mistaken for optimizer steps.

Final comparison generated at
`scripts/analysis/QK_Combination_Analysis/results/final-comparison-20260913`:
6 matched runs,28 validation points,11 LM sources,156 separate downstream
metrics,5 SVGs and SHA256 source provenance. Four comparison unit tests pass;
all5 SVGs parse. Read `scripts/analysis/QK_Combination_Analysis/FINAL_RESULTS_20260913.md`.
Last validation macro CE: L2.738568, QK2.739049, Q2.739864, K2.741134,
O2.745838, V2.746239. QK beats L at23/28 intermediate validation points, but
not the final logged validation. Positive final interaction0.003380 does not
establish a final gain over L. Single-seed and restart limitations are documented.
No extra GPU analysis/sweep was submitted. Full training, final save, terminal
success, W&B synchronization and final comparison are now all verified complete.
