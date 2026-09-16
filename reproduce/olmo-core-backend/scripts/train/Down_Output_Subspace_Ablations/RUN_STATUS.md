# Down output-subspace training status

## Submission: 2026-09-15 (Asia/Hong_Kong)

User authorized training submission and continued monitoring. Used designated
SSH gateway, conda activate MoSE, gpu_run arch 2 8 for each of the three variants.
No hyperparameter changes, GPU allocation escalation, or automatic restart enabled.
Before submission, rjob list filtered for MoSE subspace/rotation jobs found no matches.

| Variant | Created time | Job ID | Script |
|---|---|---|---|
| Identity | 22:03:49 | mose-down-subspace-identity-2x8-0915-28758665 | run_down_identity_fp8.sh |
| Static | 22:04:05 | mose-down-subspace-static-2x8-0915-44749861 | run_down_static_fp8.sh |
| Dynamic | 22:04:20 | mose-down-subspace-dynamic-2x8-0915-59799552 | run_down_dynamic_fp8.sh |

Launchers under remote personal repository:
`/mnt/shared-storage-gpfs2/intern-pretrain-shared02/shijiayang-p/archspace-MoSE/reproduce/olmo-core-backend/scripts/train/Down_Output_Subspace_Ablations/`.

22:04:32 read-only scheduler query: all three Inqueue, each2 replicas,
active0,succeeded0,failed0,pending0. Submission is confirmed; GPU startup,
FP8/FSDP/compile execution, CE, and checkpoint saves are not yet verified.
No W&B offline segment exists in the observed logs yet; do not guess sync paths.

## Continuous local monitor

Read-only checks22:05–22:11: all three still Inqueue,active0,failed0.
Verified noninteractive SSH query using zsh -ic 'conda activate MoSE && ...rjob list...'.
Started local monitor_jobs.py --daemon (PID1453363). Independent heartbeats
22:12:27 and22:13:33 HK confirm it continues querying after launch.
A duplicate monitor launch was rejected by the lock; no duplicate training submission.

Monitor artifacts (git ignored): monitor-output/status.json,history.jsonl,
monitor.log, and per-variant selected logs once replicas run. The user changed
the cadence at22:21 HK to every600s (10 minutes). Running-job logs are read at
most once per polling round. Queries retry on
transport errors without interpreting them as task failure. Stops after all
three jobs are terminal with zero active replicas and terminal log retrieval.
No automatic cancellation, re-submission, parameter changes, or W&B uploads.
Terminal segments must be resolved and synchronized before any authorized restart.
This is a local monitoring process, not a chat notification service; its continued
operation depends on this local machine and SSH access remaining available.

The earlier PID1453363 was verified no longer alive before starting the
600-second instance; no training task was stopped or altered.

22:22 HK: externally verified old PID1453363 was exactly the 60-second
monitor, terminated it with SIGTERM, and launched PID1460890 with explicit
`--interval 600`. PID command line and first heartbeat22:22:30 were verified.
All three training jobs remained Inqueue,active0 at that heartbeat. The next
normal check is approximately22:32:30 plus SSH/query latency.

## Static infrastructure restart: 2026-09-15 23:34–23:38 HKT

The 23:34:40 scheduler heartbeat found Identity and Dynamic still Running with
two active replicas each, while the original Static job
`mose-down-subspace-static-2x8-0915-44749861` was Stopped with zero active
replicas. Its terminal logs show multiple DataLoader workers killed by
`SIGTERM` at 23:33:34 on both nodes; the last observed Static train CE was
3.014. No model NaN or CUDA OOM was present in the selected terminal lines.

The exact old W&B segment
`offline-run-20260915_225138-olmo3-1b-down-output-subspace-static-g32-a05-fp8-30b`
was passed to `wandb_sjy` from the required `MoSE` environment. The command
returned zero without a PID or completion message, so invocation is confirmed
but upload completion is not independently verified.

After confirming the old job was terminal, Static was resubmitted unchanged on
`arch` with 2 nodes x 8 GPUs using the same `run_down_static_fp8.sh` launcher.
The replacement job ID is
`mose-down-subspace-static-r2-2x8-0915-57046055`; its first observed state at
23:38:24 was Inqueue with zero active replicas. The unchanged save folder is
expected to use OLMo's save-folder auto-resume, but actual checkpoint loading
and resumed optimizer/data-loader steps remain to be verified from runtime
logs.

The monitor job mapping was updated locally to the replacement Static ID. Old
monitor PID1460890 was gracefully stopped, and PID1496603 was started with
`--interval 600`. Its first heartbeat at23:38:24 verified active monitoring of
Identity, replacement Static, and Dynamic.

## Identity/Dynamic infrastructure restarts: 2026-09-16 00:40–00:45 HKT

The 00:40:58 heartbeat found both original Identity and Dynamic jobs Stopped
with zero active replicas; Static-r2 remained Running. Terminal logs for both
stopped jobs again show DataLoader workers killed by external `SIGTERM` across
their nodes, with no selected model NaN or CUDA OOM evidence.

Both runs had durable 25% checkpoints before termination. Identity successfully
saved step3576 at00:25:28 and Dynamic successfully saved step3576 at00:29:23.
Their exact original W&B offline-run segments were passed separately to
`wandb_sjy` from the required `MoSE` environment; both invocations returned
zero without a PID/completion message, so upload invocation is confirmed but
completion is not independently verified.

The unchanged Identity and Dynamic launchers were resubmitted on `arch`, each
with2 nodes x8 GPUs. Replacement IDs are
`mose-down-subspace-identity-r2-2x8-0916-39459674` and
`mose-down-subspace-dynamic-r2-2x8-0916-35036741`. Both were Inqueue with zero
active replicas at their first00:44:53 heartbeat; actual step3576 loading and
resumed training remain to be verified in runtime logs.

The monitor mapping was updated to both replacements. PID1496603 was gracefully
stopped and PID1510093 launched with `--interval 600`; its first heartbeat
verified active monitoring of Identity-r2, Static-r2, and Dynamic-r2.

## Static-r2 infrastructure restart: 2026-09-16 01:05–01:09 HKT

At01:05:21 Static-r2 was Stopped with zero active replicas while Identity-r2
and Dynamic-r2 remained queued. Its terminal logs show DataLoader worker
`SIGTERM` errors across both nodes followed by expected Gloo peer-connection
failures; no selected model NaN or CUDA OOM evidence was present.

The exact Static-r2 W&B segment was resolved as
`offline-run-20260915_234309-olmo3-1b-down-output-subspace-static-g32-a05-fp8-30b`
and passed to `wandb_sjy`; invocation returned zero without independent upload
completion evidence. Static-r2 successfully saved step1788 at00:32:40.

Static was resubmitted unchanged on `arch`2x8 as
`mose-down-subspace-static-r3-2x8-0916-75612222`, expected to auto-resume from
step1788, subject to runtime verification. The monitor mapping was updated;
PID1510093 was gracefully replaced by PID1514898 with `--interval 600`.
Its first heartbeat at01:08:37 showed Identity-r2, Static-r3, and Dynamic-r2
all Inqueue with zero active replicas.

## Identity-r2/Dynamic-r2 resume verification: 2026-09-16 02:08–02:12 HKT

The 02:09:15 heartbeat found Identity-r2 and Dynamic-r2 Running with two
active replicas each; Static-r3 remained Inqueue. Runtime logs explicitly
reported `Will resume training from step 3576, epoch 1` for both replacement
jobs, so neither restarted from step zero. Dynamic-r2 subsequently emitted
new train CE values (2.771, 2.782, and 2.818), confirming resumed optimizer
steps rather than checkpoint loading alone.

The new W&B offline segments are:

- Identity-r2: `offline-run-20260916_020918-olmo3-1b-down-output-subspace-identity-g32-a05-fp8-30b`
- Dynamic-r2: `offline-run-20260916_020804-olmo3-1b-down-output-subspace-dynamic-g32-a05-fp8-30b`

These replacement segments have not been synchronized because both jobs are
still active. Static-r3 is expected to resume from its durable step1788
checkpoint once scheduled; this remains pending runtime verification.

## Static-r3 resume verification: 2026-09-16 02:12–02:20 HKT

The 02:19:36 heartbeat found all three variants Running with two active
replicas each. Static-r3 logs explicitly reported loading
`stage1/step1788` and `Will resume training from step 1788, epoch 1`.
It completed the resumed validation pass and emitted finite train CE values
(3.011, 2.963, 2.981, 3.030, and 2.997 in the sampled lines), with no NaN or
CUDA OOM evidence.

Static-r3's new, currently active W&B segment is
`offline-run-20260916_021249-olmo3-1b-down-output-subspace-static-g32-a05-fp8-30b`.
It has not been synchronized because the replacement job is still active.

At03:00:46 Static-r3 began its asynchronous step3576 save, and at03:00:50
rank0 reported `Checkpoint for step 3576 saved successfully`. The03:01:25
heartbeat still showed all three jobs Running with two active replicas each.

At03:13 HKT a read-only checkpoint-directory audit confirmed durable latest
steps: Identity step5364, Static step3576, and Dynamic step5364. The03:11:52
heartbeat had all three jobs Running with two active replicas each, and the
sampled validation/train CE values were finite with no NaN/OOM/SIGTERM lines.

At03:54 HKT another read-only checkpoint audit confirmed Identity and Dynamic
had reached durable step7152 (50%), while Static had reached durable step5364
(37.5%). The03:53:43 heartbeat still showed all three jobs Running with two
active replicas each; sampled CE remained finite and no fault keywords were
present.

At04:25:13 Identity-r2 began its asynchronous step8940 save (62.5%), and
rank0 reported successful completion at04:25:17. The04:25:05 scheduler
heartbeat still showed all three jobs Running with two active replicas each.

At04:34:51 Static-r3 began saving step7152 (50%), and rank0 reported
successful completion at04:34:55. The04:35:33 heartbeat still showed all
three jobs Running with two active replicas each. Dynamic-r2 completed a
finite validation pass at04:34:45 (including C4 CE3.239 and Pile CE2.481),
with no fault keywords observed.

At04:37 HKT a read-only directory audit confirmed Dynamic-r2 had completed
its durable step8940 checkpoint (62.5%).

At05:28 HKT a read-only checkpoint audit confirmed Identity and Dynamic at
durable step10728 (75%), and Static at durable step8940 (62.5%). The05:27:50
heartbeat showed all three jobs Running with two active replicas each; sampled
train CE remained finite and no fault keywords were present.

At06:08:52 Static-r3 began saving step10728 (75%), and rank0 reported a
successful save at06:08:55. The06:09:41 heartbeat showed all three jobs
Running with two active replicas each. Identity completed another finite
validation pass and Dynamic emitted finite MMLU fast-eval CE values; no fault
keywords were observed.

At06:12 HKT a read-only checkpoint audit confirmed Identity-r2 and Dynamic-r2
had reached durable step12516 (87.5%), while Static-r3 remained at durable
step10728 (75%). All three jobs were still Running with two active replicas
each, with finite sampled train/validation CE and no fault keywords observed.

Identity-r2 saved its periodic step14304 checkpoint at06:42:07 and its final
step14306 checkpoint immediately afterward. Rank0 logged `Training complete`
at06:42:16, and the06:51:26 scheduler heartbeat reported the job Succeeded.
Its exact r2 W&B offline segment was then synced with `wandb_sjy`; the command
returned exit status0. Static-r3 and Dynamic-r2 remained Running with two
active replicas each.

Dynamic-r2 saved its periodic step14304 checkpoint at06:58:08 and final
step14306 checkpoint at06:58:14, then logged `Training complete` at06:58:18.
The07:01:55 scheduler heartbeat reported Dynamic-r2 Succeeded. Its exact r2
W&B offline segment was synced with `wandb_sjy`, which returned exit status0.
Static-r3 remained Running with two active replicas.

At07:12 HKT the scheduler still reported Static-r3 Running with two active
replicas, and a read-only checkpoint audit confirmed durable step12516
(87.5%). Its sampled train CE remained finite and no fault keywords appeared.

Static-r3 saved its periodic step14304 checkpoint at07:44:01 and final
step14306 checkpoint at07:44:04, then logged `Training complete` at07:44:11.
The07:53:06 scheduler heartbeat reported all three variants Succeeded and the
monitor wrote its terminal snapshot. A final directory audit confirmed
Static-r3 step10728, step12516, step14304, and step14306. Its exact r3 W&B
offline segment was synced with `wandb_sjy`, which returned exit status0.

Final state: Identity-r2, Static-r3, and Dynamic-r2 all completed the full
30B-token schedule, all three have durable final step14306 checkpoints, and
all three current offline W&B segments have been synchronized.

## Dynamic residual-gate follow-up: 2026-09-16 13:40 HKT

Added a backward-compatible `dynamic_residual` mode:
`y_out = y + tanh(g) * (R(z)y - y)`. The 32 per-group gates are initialized
to exactly zero, while the rotation controller uses an RNG-isolated nonzero
normal initialization with standard deviation `1/sqrt(rank)`. Consequently
step 0 is exactly the Identity forward, the first backward updates the gate,
and the controller begins receiving gradients after the gate leaves zero.
The completed `dynamic` experiment and its checkpoint format were not changed.

Local verification passed all 10 output-subspace unit tests, including exact
Identity start, staged gradients, config values, meta initialization, compile,
optimizer categorization, and Float8 conversion. Nine Down-Identity/All-Linear
regression tests also passed; the separate pytest-dependent test module was not
run because pytest is absent from the local environment.

Submitted on `arch` with 2 nodes x 8 GPUs:

- Scheduler job: `mose-down-dyn-resgate-29251295`
- Run name: `olmo3-1b-down-output-subspace-dynamic-residual-g32-a05-fp8-30b`
- Launcher: `run_down_dynamic_residual_fp8.sh`
- Schedule: 30B tokens, global BS256, seq8192, Muon LR0.005, FP8, WSD warmup
  2000 steps / final 20% decay, checkpoint interval 1788 steps.

Submission was accepted by `gpu_run`; training start/completion has not yet
been verified.

At 2026-09-16 13:51 HKT, a read-only scheduler check reported the job
`Inqueue` with 0/2 active replicas, 0 failed replicas, and 0 created-failed
replicas. It has not started training yet, and there is no failure to diagnose.

At 18:58 HKT, the job had reached log step 493/14306 and both replicas were
then stopped by SIGTERM. A 19:17 scheduler check confirmed `Stopped`, with
0 active and 2 stopped replicas. DataLoader-worker termination and NCCL broken
pipe messages followed SIGTERM; the source of the stop was not established.
Step 493 is before the first periodic checkpoint at step 1788.

On explicit user request to resubmit, invoked `wandb_sjy` for the exact old
segment
`/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20260916_184444-olmo3-1b-down-output-subspace-dynamic-residual-g32-a05-fp8-30b`.
The command returned exit status 0 with no upload-completion output, so online
sync completion remains unverified. The replacement was then submitted to
`arch` with 2 x 8 GPUs, unchanged launcher/run name/stage/checkpoint path:

- New scheduler job: `mose-down-dyn-resgate-r2-90213700`
- 2026-09-16 19:18 HKT scheduler check: `Inqueue`, 0/2 active, 0 failed.
- Resume step is not yet verified from replacement logs.
