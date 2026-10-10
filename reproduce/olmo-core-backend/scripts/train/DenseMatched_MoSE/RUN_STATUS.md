# Dense-matched MoSE run status

## Submission: 2026-09-29 00:15 HKT

Both requested experiments were submitted through the designated gateway with
gpu_run on the arch partition. Each uses 2 nodes x 8 GPUs and low priority
(PRIORITY=1). Automatic scheduler restart is disabled.

### F0: unshared bottleneck control

- Actual scheduler job:
  mose-dm-f0-r1-p1-2x8-0929-a7c31-24291007
- Launcher:
  scripts/train/DenseMatched_MoSE/run_f0_unshared.sh
- W&B run name:
  olmo3-1b-densematched-mose-f0-unshared-down-rms-fp8-30b
- Submission response confirmed that the RJob was created.

### F1: shared K/V and Gate/Up

- Actual scheduler job:
  mose-dm-f1-r1-p1-2x8-0929-b4d62-31557804
- Launcher:
  scripts/train/DenseMatched_MoSE/run_f1_kv_gateup_shared.sh
- W&B run name:
  olmo3-1b-densematched-mose-f1-kv-gateup-shared-down-rms-fp8-30b
- Submission response confirmed that the RJob was created.

No scheduler status or logs were queried as part of this submission request.
Created does not establish that either job has started training.

## Monitoring: 2026-09-29 00:36 HKT

An authorized scheduler query in namespace `ailab-llmarchitecture` confirmed
that both exact jobs remain `Inqueue`. Each still has two replicas with
`active=0`, `succeeded=0`, `failed=0`, `created_failed=0`, `pending=0`, and
`stopped=0`; both replica placeholders are `STARTING` without assigned hosts.
This is queue waiting rather than a stopped attempt, so no W&B synchronization
or replacement submission was triggered. The next normal check is due around
01:06 HKT under the requested 30-minute reporting cadence.

## Monitoring: 2026-09-29 01:07 HKT

The original SSH connection expired during the interval; this was treated as
a transport event rather than a job failure. After reconnecting, an
authoritative query of both exact handles again showed `Inqueue`, with two
`STARTING` replicas per job and all active/success/failure/pending/stopped
counters at zero. Neither attempt was duplicated or restarted. The next normal
check is due around 01:37 HKT.

## Monitoring: 2026-09-29 01:38 HKT

Both exact jobs remain `Inqueue`, with two `STARTING` replicas each and every
activity/terminal counter at zero. The interval SSH connection again expired;
a fresh authenticated scheduler query produced the status above. No W&B sync,
restart, or duplicate submission was needed. The next normal check is due
around 02:08 HKT.

## Monitoring: 2026-09-29 02:09 HKT

Both exact handles are still `Inqueue`. Each has two `STARTING` placeholders,
no assigned host, and zero active/succeeded/failed/created-failed/pending/stopped
replicas. The expired interval SSH session was replaced before making this
authoritative query. No restart or upload was triggered. The next check is due
around 02:39 HKT.

## Monitoring: 2026-09-29 02:40 HKT

The two original jobs remain `Inqueue`. Both continue to show two unassigned
`STARTING` replicas and zero active or terminal counters. No stop condition was
observed, so neither job was synchronized or resubmitted. The next check is due
around 03:10 HKT.

## Monitoring: 2026-09-29 03:10 HKT

Both exact handles remain `Inqueue`, each with two unassigned `STARTING`
replicas and all activity/terminal counters at zero. No W&B sync or replacement
submission was required. The next check is due around 03:40 HKT.

## Monitoring: 2026-09-29 03:41 HKT

Both jobs remain `Inqueue`, each with two unassigned `STARTING` replicas and
zero active or terminal counters. No synchronization or restart was triggered.
The next check is due around 04:11 HKT.

## Monitoring: 2026-09-29 04:11 HKT

Both jobs remain `Inqueue`, each with two unassigned `STARTING` replicas and
zero active or terminal counters. No synchronization or restart was triggered.
The next check is due around 04:41 HKT.

## Monitoring: 2026-09-29 04:42 HKT

Both jobs remain `Inqueue`, each with two unassigned `STARTING` replicas and
zero active or terminal counters. No synchronization or restart was triggered.
The next check is due around 05:12 HKT.

## Monitoring: 2026-09-29 05:12 HKT

Both jobs remain `Inqueue`, each with two unassigned `STARTING` replicas and
zero active or terminal counters. No synchronization or restart was triggered.
The next check is due around 05:42 HKT.

## Monitoring: 2026-09-29 05:43 HKT

Both jobs remain `Inqueue`, each with two unassigned `STARTING` replicas and
zero active or terminal counters. No synchronization or restart was triggered.
The next check is due around 06:13 HKT.

## Monitoring: 2026-09-29 06:13 HKT

Both jobs remain `Inqueue`, each with two unassigned `STARTING` replicas and
zero active or terminal counters. No synchronization or restart was triggered.
The next check is due around 06:43 HKT.

## Monitoring: 2026-09-29 06:44 HKT

Both jobs remain `Inqueue`, each with two unassigned `STARTING` replicas and
zero active or terminal counters. No synchronization or restart was triggered.
The next check is due around 07:14 HKT.

## Monitoring: 2026-09-29 07:14 HKT

Both jobs remain `Inqueue`, each with two unassigned `STARTING` replicas and
zero active or terminal counters. No synchronization or restart was triggered.
The next check is due around 07:44 HKT.

## Monitoring: 2026-09-29 07:44 HKT

Both jobs remain `Inqueue`, each with two unassigned `STARTING` replicas and
zero active or terminal counters. No synchronization or restart was triggered.
The next check is due around 08:14 HKT.

## Monitoring: 2026-09-29 08:15 HKT

Both jobs remain `Inqueue`, each with two unassigned `STARTING` replicas and
zero active or terminal counters. No synchronization or restart was triggered.
The next check is due around 08:45 HKT.

## Monitoring: 2026-09-29 08:45 HKT

Both jobs remain `Inqueue`, each with two unassigned `STARTING` replicas and
zero active or terminal counters. No synchronization or restart was triggered.
The next check is due around 09:15 HKT.

## Monitoring: 2026-09-29 09:16 HKT

Both jobs remain `Inqueue`, each with two unassigned `STARTING` replicas and
zero active or terminal counters. No synchronization or restart was triggered.
The next check is due around 09:46 HKT.

## Monitoring: 2026-09-29 09:46 HKT

Both jobs remain `Inqueue`, each with two unassigned `STARTING` replicas and
zero active or terminal counters. No synchronization or restart was triggered.
The next check is due around 10:16 HKT.

## Monitoring: 2026-09-29 10:17 HKT

Both jobs remain `Inqueue`, each with two unassigned `STARTING` replicas and
zero active or terminal counters. No synchronization or restart was triggered.
The next check is due around 10:47 HKT.

## Monitoring: 2026-09-29 10:47 HKT

Both jobs remain `Inqueue`, each with two unassigned `STARTING` replicas and
zero active or terminal counters. No synchronization or restart was triggered.
The next check is due around 11:17 HKT.

## Monitoring: 2026-09-29 11:18 HKT

Both jobs remain `Inqueue`, each with two unassigned `STARTING` replicas and
zero active or terminal counters. No synchronization or restart was triggered.
The next check is due around 11:48 HKT.

## Monitoring: 2026-09-29 11:48 HKT

Both jobs remain `Inqueue`, each with two unassigned `STARTING` replicas and
zero active or terminal counters. The full observation history since submission
still contains no allocation or running interval. No synchronization or restart
was triggered. The next check is due around 12:18 HKT.

## Monitoring: 2026-09-29 12:19 HKT

Both jobs remain `Inqueue`, each with two unassigned `STARTING` replicas and
zero active or terminal counters. No synchronization or restart was triggered.
The next check is due around 12:49 HKT.

## Monitoring: 2026-09-29 12:49 HKT

Both jobs remain `Inqueue`, each with two unassigned `STARTING` replicas and
zero active or terminal counters. No synchronization or restart was triggered.
The next check is due around 13:19 HKT.

## Monitoring: 2026-09-29 13:20 HKT

Both jobs remain `Inqueue`, each with two unassigned `STARTING` replicas and
zero active or terminal counters. No synchronization or restart was triggered.
The next check is due around 13:50 HKT.

## Monitoring: 2026-09-29 13:54 HKT

The prior monitoring wait was interrupted by a session/tool reset; this was not
treated as job failure. A fresh authoritative query confirms both jobs remain
`Inqueue`, each with two unassigned `STARTING` replicas and zero active or
terminal counters. No synchronization or restart was triggered. The next check
is due around 14:24 HKT.

## Monitoring: 2026-09-29 14:24 HKT

Both jobs remain `Inqueue`, each with two unassigned `STARTING` replicas and
zero active or terminal counters. No synchronization or restart was triggered.
The next check is due around 14:54 HKT.

## Monitoring: 2026-09-29 15:41 HKT

The scheduled wait session was interrupted before its result could be read. A
fresh authoritative query confirms both exact jobs remain `Inqueue`, each with
two unassigned `STARTING` replicas and zero active or terminal counters. The
transport/tool interruption was not treated as task failure; no synchronization
or restart was triggered. The next check is due around 16:11 HKT.

## Monitoring: 2026-09-29 16:11–16:14 HKT

F0 received two H200 nodes and is now `Running` with both replicas active;
F1 remains `Inqueue` with two unassigned `STARTING` replicas and all counters
at zero. F0 runtime logs confirm all 16 ranks, data-parallel world size 16, and
the requested `(dp_replicate=2, dp_shard=8)` mesh. It found no prior checkpoint,
saved `stage1/step0`, completed the forward/backward dry run, and began epoch 1.
The exact offline W&B segment is:

`/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20260929_161226-olmo3-1b-densematched-mose-f0-unshared-down-rms-fp8-30b`

Step 1 reported CE 11.90 and total grad norm 2.810; step 10 CE 11.45 and
grad norm 2.751; step 20 CE 10.28 and grad norm 2.373, with about 57.3k
tokens/s/device and 56.1% MFU. Logs reached step 27 without OOM, NaN, traceback,
or fatal error. The ordinary synchronized-CUDA warnings are non-terminal.
No restart or W&B synchronization was triggered. The next check is due around
16:44 HKT.

## Monitoring: 2026-09-29 16:45–16:46 HKT

F0 remains `Running` with both replicas active on the same two H200 nodes and
advanced through step 800. Step 800 reported CE 3.193 and total grad norm
0.1592; no OOM, NaN, traceback, or runtime error appeared in the selected log
tail. F1 remains `Inqueue` with two unassigned `STARTING` replicas and all
counters at zero. No synchronization or restart was triggered. The next check
is due around 17:16 HKT.

## Monitoring: 2026-09-29 17:16 HKT

F0 remains `Running` with both replicas active and advanced through step 1539.
Step 1530 reported CE 2.998 and total grad norm 0.1135; no OOM, NaN,
traceback, or runtime error appeared in the selected log tail. F1 remains
`Inqueue` with two unassigned `STARTING` replicas and all counters at zero. No
synchronization or restart was triggered. The next check is due around 17:46
HKT.

## Monitoring: 2026-09-30 00:24 HKT

After the previous monitoring wait was interrupted, a fresh authoritative
query confirms F0 is still `Running` with both replicas active on its original
two H200 nodes. It has advanced through step 11982/14306; step 11980 reported
CE 2.538 and total grad norm 0.0229, with an ETA of about 1h36m. The selected
recent log tail contains no OOM, NaN, traceback, or runtime error. F1 remains
`Inqueue` with two unassigned `STARTING` replicas, zero active replicas, and no
terminal counters. Neither task has stopped, so no synchronization or restart
was triggered. The next check is due around 00:54 HKT.

## Monitoring: 2026-09-30 00:55 HKT

Both jobs are now `Running` with two active replicas each. F0 remains on its
original H200 nodes and advanced through step 12692/14306. Step 12690 reported
CE 2.469 and total grad norm 0.0176, with an ETA of about 1h07m.

F1 received two H200 nodes at about 00:35 HKT. Runtime logs confirm 16 ranks
across the two nodes and the requested `(dp_replicate=2, dp_shard=8)` device
mesh. It found no prior checkpoint, saved `stage1/step0` successfully at
00:38:02, completed the forward/backward dry run at 00:38:30, and began
training from scratch. Its exact offline W&B segment is:

`/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20260930_003803-olmo3-1b-densematched-mose-f1-kv-gateup-shared-down-rms-fp8-30b`

F1 step 1 reported CE 11.90 and total grad norm 2.759. It advanced through
step 476; step 470 reported CE 3.453 and total grad norm 0.2284. Neither job's
selected logs show OOM, NaN, traceback, or runtime error. No synchronization
or restart was triggered. The next check is due around 01:25 HKT.

## Monitoring: 2026-09-30 01:27 HKT

Both jobs remain `Running` with two active replicas each. F0 advanced through
step 13469/14306; step 13460 reported CE 2.424 and total grad norm 0.0120,
with an ETA of about 35 minutes. F1 advanced through step 1270/14306; step
1270 reported CE 2.985 and total grad norm 0.1805. The selected recent log
tails contain no OOM, NaN, traceback, or runtime error. No synchronization or
restart was triggered. The next check is due around 01:57 HKT.

## Monitoring: 2026-09-30 01:57 HKT

Both jobs remain `Running` with two active replicas each. F0 advanced through
step 14165/14306; step 14160 reported CE 2.335 and total grad norm 0.0081,
with an ETA of about six minutes. F1 advanced through step 2084/14306; step
2080 reported CE 2.842 and total grad norm 0.0684, with an ETA of about 7h40m.
The selected recent log tails contain no OOM, NaN, traceback, or runtime error.
No synchronization or restart was triggered. The next check is due around
02:27 HKT.

## Monitoring: 2026-09-30 02:28 HKT

F0 completed successfully. The trainer reached step 14306/14306, initiated the
final `stage1/step14306` checkpoint save, finalized the W&B run, logged
`Training complete` at 02:02:51, and the scheduler now reports `Succeeded`
with both replicas succeeded and zero active/failed/stopped replicas. Its exact
offline W&B segment was submitted with `wandb_sjy` (background PID 677729), and
`/tmp/wandb_sjy.log` confirms `Finished syncing` through history step 14306.
No restart is required for this completed job.

F1 remains `Running` with both replicas active and advanced through step
2938/14306. Step 2930 reported CE 2.762 and total grad norm 0.0325, with an ETA
of about 7h02m. The selected recent log tail contains no OOM, NaN, traceback,
or runtime error. The next check is due around 02:58 HKT.

## Monitoring: 2026-09-30 03:01 HKT

F0 remains authoritatively `Succeeded` with both replicas succeeded. F1
remains `Running` with both replicas active and advanced through step
3759/14306. Step 3750 reported CE 2.636 and total grad norm 0.0501, with an ETA
of about 6h38m. The selected recent log tail contains no OOM, NaN, traceback,
or runtime error. No synchronization or restart was triggered. The next check
is due around 03:31 HKT.

## Monitoring: 2026-09-30 03:31 HKT

F0 remains authoritatively `Succeeded` with both replicas succeeded. F1
remains `Running` with both replicas active and advanced through step
4605/14306. Step 4600 reported CE 2.648 and total grad norm 0.0262, with an ETA
of about 6h03m. The selected recent log tail contains no OOM, NaN, traceback,
or runtime error. No synchronization or restart was triggered. The next check
is due around 04:01 HKT.

## Monitoring: 2026-09-30 04:02 HKT

F0 remains authoritatively `Succeeded` with both replicas succeeded. F1
remains `Running` with both replicas active and advanced through step
5471/14306. Step 5470 reported CE 2.593 and total grad norm 0.0204, with an ETA
of about 5h27m. The selected recent log tail contains no OOM, NaN, traceback,
or runtime error. No synchronization or restart was triggered. The next check
is due around 04:32 HKT.

## Monitoring: 2026-09-30 04:32 HKT

F0 remains authoritatively `Succeeded` with both replicas succeeded. F1
remains `Running` with both replicas active and advanced through step
6317/14306. Step 6310 reported CE 2.640 and total grad norm 0.0226, with an ETA
of about 4h55m. The selected recent log tail contains no OOM, NaN, traceback,
or runtime error. No synchronization or restart was triggered. The next check
is due around 05:02 HKT.

## Monitoring: 2026-09-30 05:02 HKT

F0 remains authoritatively `Succeeded` with both replicas succeeded. F1
remains `Running` with both replicas active and advanced through step
7160/14306. Step 7160 reported CE 2.610 and total grad norm 0.0240, with an ETA
of about 4h23m. Its scheduled `stage1/step7152` checkpoint was saved
successfully at 05:02:02. The selected recent log tail contains no OOM, NaN,
traceback, or runtime error. No synchronization or restart was triggered. The
next check is due around 05:32 HKT.

## Monitoring: 2026-09-30 05:34 HKT

The scheduled monitoring shell was no longer available, so a fresh
authoritative scheduler query was used; this transport/session loss was not
treated as job failure. F0 remains `Succeeded`. F1 remains `Running` with both
replicas active and advanced through step 8033/14306. Step 8030 reported CE
2.598 and total grad norm 0.0143, with an ETA of about 3h51m. Step 8000's
periodic LM evaluation also completed normally. The selected recent log tail
contains no OOM, NaN, traceback, or runtime error. No synchronization or
restart was triggered. The next check is due around 06:04 HKT.

## Monitoring: 2026-09-30 06:04 HKT

F0 remains authoritatively `Succeeded` with both replicas succeeded. F1
remains `Running` with both replicas active and advanced through step
8898/14306. Step 8890 reported CE 2.504 and total grad norm 0.0230, with an ETA
of about 3h18m. The selected recent log tail contains no OOM, NaN, traceback,
or runtime error. No synchronization or restart was triggered. The next check
is due around 06:34 HKT.

## Monitoring: 2026-09-30 06:34 HKT

F0 remains authoritatively `Succeeded` with both replicas succeeded. F1
remains `Running` with both replicas active and advanced through step
9744/14306. Step 9740 reported CE 2.582 and total grad norm 0.0212, with an ETA
of about 2h47m. The selected recent log tail contains no OOM, NaN, traceback,
or runtime error. No synchronization or restart was triggered. The next check
is due around 07:04 HKT.

## Monitoring: 2026-09-30 07:05 HKT

F0 remains authoritatively `Succeeded` with both replicas succeeded. F1
remains `Running` with both replicas active and advanced through step
10618/14306. Step 10610 reported CE 2.506 and total grad norm 0.0149, with an
ETA of about 2h15m. The selected recent log tail contains no OOM, NaN,
traceback, or runtime error. No synchronization or restart was triggered. The
next check is due around 07:35 HKT.

## Monitoring: 2026-09-30 07:36 HKT

F0 remains authoritatively `Succeeded` with both replicas succeeded. F1
remains `Running` with both replicas active and advanced through step
11489/14306. Step 11480 reported CE 2.502 and total grad norm 0.0173, with an
ETA of about 1h43m. The selected recent log tail contains no OOM, NaN,
traceback, or runtime error. No synchronization or restart was triggered. The
next check is due around 08:06 HKT.

## Monitoring: 2026-09-30 08:06 HKT

F0 remains authoritatively `Succeeded` with both replicas succeeded. F1
remains `Running` with both replicas active and advanced through step
12333/14306. Step 12330 reported CE 2.436 and total grad norm 0.0116, with an
ETA of about 1h12m. The selected recent log tail contains no OOM, NaN,
traceback, or runtime error. No synchronization or restart was triggered. The
next check is due around 08:36 HKT.

## Monitoring: 2026-09-30 08:36 HKT

F0 remains authoritatively `Succeeded` with both replicas succeeded. F1
remains `Running` with both replicas active and advanced through step
13141/14306. Step 13140 reported CE 2.441 and total grad norm 0.0159, with an
ETA of about 43 minutes. The selected recent log tail contains no OOM, NaN,
traceback, or runtime error. No synchronization or restart was triggered. The
next check is due around 09:06 HKT.

## Monitoring: 2026-09-30 09:07 HKT

F0 remains authoritatively `Succeeded` with both replicas succeeded. F1
remains `Running` with both replicas active and advanced through step
13999/14306. Step 13990 reported CE 2.443 and total grad norm 0.0099, with an
ETA of about 12 minutes. The selected recent log tail contains no OOM, NaN,
traceback, or runtime error. No synchronization or restart was triggered. The
next check is due around 09:37 HKT.

## Completion: 2026-09-30 09:37 HKT

Both experiments are authoritatively `Succeeded` with zero active replicas
and two succeeded replicas each. F1 reached step 14306/14306, initiated its
final `stage1/step14306` checkpoint, finalized the W&B run, and logged
`Training complete` at 09:18:40 HKT. Its exact offline W&B directory was
synced with `wandb_sjy`, and `/tmp/wandb_sjy.log` confirmed `Finished
syncing` through history step 14306. F0 had already completed and its exact
offline W&B directory had likewise been synced successfully. No restart was
needed for either run.
