# Rank-512-budget combination run status

## 2026-10-01 22:44 HKT — C7 submitted

- Added C7 to isolate the effect of SiLU between both shared Gate/Up factors. Relative to C6, Q/K SiLU, V/O Identity, Gate/Up sharing, Down RMSNorm, ranks, seed, and 30B FP8 recipe are unchanged.
- Validated Python and Bash syntax without writing bytecode; the per-block projection budget remains exactly 24,117,248 parameters, matching the all-unshared rank-512 reference.
- Submitted to the non-pool `arch` allocation (`ailab-llmarchitecture`, charged group `llmarchitecture_gpu`) at low priority 1, 2 nodes x 8 GPUs. Scheduler created `mose-r512-c7-r1-p1-2x8-np-1001-2246-59542618`; auto restart is disabled.
- Launcher: `run_c7_qk_silu_ug_silu_down_rms_gateup.sh`. W&B run name: `olmo3-1b-r512matched-c7-qk-silu-ug-silu-down-rms-gateup-fp8-30b`.
- Startup, first optimizer step, and completion are not yet confirmed. Compare its last-1000-step train CE against C6 (2.474763, with step 12516 anomalous row excluded consistently).

## 2026-10-02 12:40 HKT — C7 completed and W&B synced

- Scheduler reports `Succeeded`, active=0, succeeded=2, failed=0, stopped=0 for `mose-r512-c7-r1-p1-2x8-np-1001-2246-59542618`.
- Logs confirm all 14,306 steps, final `stage1/step14306` checkpoint, and `Training complete` at 05:43:20 HKT.
- Exact W&B segment: `/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261001_224653-olmo3-1b-r512matched-c7-qk-silu-ug-silu-down-rms-gateup-fp8-30b`.
- Invoked `wandb_sjy` for that exact segment and verified `Finished syncing` in `/tmp/wandb_sjy.log`. W&B run: `https://wandb.ai/archspace/MoSE/runs/olmo3-1b-r512matched-c7-qk-silu-ug-silu-down-rms-gateup-fp8-30b`.
- Matched final-1000-step Train CE comparison with C6 is pending full-history analysis; one logged step is not a meaningful comparison.

## Submission: 2026-09-30 13:32--13:33 HKT

Both primary Train-CE candidates were submitted to `arch` with 2 nodes x 8
GPUs and low priority (`PRIORITY=1`). Scheduler auto-restart is disabled.

| Run | Scheduler job | Launcher |
| --- | --- | --- |
| C1 | `mose-r512-c1-r1-p1-2x8-0930-65936346` | `run_c1_k_silu_down_rms_kv.sh` |
| C3 | `mose-r512-c3-r1-p1-2x8-0930-152403` | `run_c3_k_silu_down_rms_kv_gateup.sh` |

The submission responses confirmed both RJobs were created.

## Monitoring: 2026-09-30 14:13--14:14 HKT

A fresh authorized scheduler query confirmed that both exact jobs are
`Inqueue`. Each has two unassigned `STARTING` replicas and all
active/succeeded/failed/created-failed/pending/stopped counters are zero. The
original interval SSH query did not return, so it was treated as a monitoring
transport issue rather than job failure. No W&B synchronization or replacement
submission was triggered. The next check is due around 14:44 HKT.

## 2026-09-30 14:53 HKT — second 30-minute check

- C1 `mose-r512-c1-r1-p1-2x8-0930-65936346`: `Running`, 2/2 active replicas, no failed/stopped replicas. At about 14:55 it had reached step 725/14306; step 720 train CE 3.404, grad norm 0.1973, about 86.1k tokens/s/device, MFU 48.39%.
- C3 `mose-r512-c3-r1-p1-2x8-0930-152403`: `Running`, 2/2 active replicas, no failed/stopped replicas. At about 14:55 it had reached step 60/14306; step 60 train CE 7.605, grad norm 0.3511, about 85.5k tokens/s/device, MFU 48.08%.
- Both log tails were advancing without OOM, NaN, or traceback. C3 started substantially later than C1.

## 2026-09-30 15:30 HKT — third 30-minute check

- C1 remains `Running`, 2/2 active, no failed/stopped replicas. Log advanced through step 2098/14306. At step 2090: train CE 2.975, grad norm 0.0803, actual-average throughput 85,522 tokens/s/device, MFU 48.07%, ETA about 5h13m.
- C3 remains `Running`, 2/2 active, no failed/stopped replicas. Log advanced through step 1372/14306. At step 1370: train CE 3.105, grad norm 0.1692, ETA about 5h48m.
- Both jobs are advancing normally; no OOM, NaN, traceback, or scheduler failure. No resubmission needed.

## 2026-09-30 16:04--16:06 HKT — fourth check and third concurrent submission

- C1 remains `Running`, 2/2 active, no failed/stopped replicas. It advanced through step 3479/14306; step 3470 train CE 2.821, grad norm 0.0532, ETA about 4h35m.
- C3 remains `Running`, 2/2 active, no failed/stopped replicas. It advanced through step 2606/14306; step 2600 train CE 2.827, grad norm 0.0705, ETA about 5h18m.
- With the user's authorization to use a third concurrent slot when capacity is available, submitted recommended third candidate C4 (all-Attention SiLU + KV sharing, Down RMSNorm), 2 nodes x 8 GPUs on `arch`, low priority `PRIORITY=1`:
  - scheduler job: `mose-r512-c4-r1-p1-2x8-0930-1605-44428407`
  - launcher: `run_c4_attention_silu_down_rms_kv.sh`
  - scheduler confirmed auto restart disabled.
- Immediate status at 16:06:21: C4 is `Running`, 2/2 active, no failed/stopped replicas, on hosts `gpu-l-lg-cmc-h-h200-0094` and `gpu-lg-cmc-h-h200-2298`.

## 2026-09-30 16:38--16:40 HKT — fifth check and fourth candidate queued

- C1 remains `Running`, 2/2 active, no failed/stopped replicas. It advanced through step 4803/14306; step 4800 train CE 2.751, grad norm 0.0335, ETA about 4h02m.
- C3 remains `Running`, 2/2 active, no failed/stopped replicas. It advanced through step 3867/14306; latest reported train CE at step 3860 was 2.704 and grad norm 0.0357.
- C4 remains `Running`, 2/2 active, no failed/stopped replicas. It advanced through step 1185/14306; step 1180 train CE 3.075, grad norm 0.2493, ETA about 5h30m.
- After the user's authorization to try additional candidates when capacity permits, submitted C0 (clean K-only SiLU + Down RMSNorm control), `arch` 2x8, `PRIORITY=1`:
  - scheduler job: `mose-r512-c0-r1-p1-2x8-0930-1639-81763768`
  - launcher: `run_c0_k_silu_down_rms.sh`
  - auto restart disabled.
- Immediate status at 16:40:18: C0 is `Inqueue`, with both replicas `STARTING` and all counters zero. Since the new 2x8 request did not receive capacity immediately, C2 was not additionally queued in this interval.

## 2026-09-30 17:12--17:13 HKT — sixth check

- C1 remains `Running`, 2/2 active, no failed/stopped replicas. It advanced through step 6141/14306; step 6140 train CE 2.659, grad norm 0.0208, ETA about 3h28m.
- C3 remains `Running`, 2/2 active, no failed/stopped replicas. It advanced past step 5140/14306; step 5140 train CE 2.664, grad norm 0.0221, ETA about 4h09m.
- C4 remains `Running`, 2/2 active, no failed/stopped replicas. It advanced through step 2541/14306; step 2540 train CE 2.877, grad norm 0.0783, ETA about 4h56m.
- C0 remains `Inqueue`, both replicas `STARTING`, with all counters zero. This is queueing, not failure. Since the pending 2x8 job still has no allocation, C2 was not submitted in this interval.

## 2026-09-30 17:45--17:48 HKT — seventh check and fifth candidate submission

- C1 remains `Running`, 2/2 active, no failed/stopped replicas. It advanced past step 7550/14306; step 7550 train CE 2.624, grad norm 0.0191, ETA about 2h52m.
- C3 remains `Running`, 2/2 active, no failed/stopped replicas. It advanced through step 6416/14306; step 6410 train CE 2.691, grad norm 0.0388, ETA about 3h33m.
- C4 remains `Running`, 2/2 active, no failed/stopped replicas. It advanced through step 3944/14306; step 3940 train CE 2.816, grad norm 0.1207, ETA about 4h19m.
- C0 acquired capacity and is now `Running`, 2/2 active, no failed/stopped replicas. It advanced through step 877/14306; step 870 train CE 3.287, grad norm 0.2109.
- Since another 2x8 allocation became available, submitted the final prepared candidate C2 (K-only SiLU + shared Gate/Up + Down RMSNorm), `arch` 2x8, `PRIORITY=1`:
  - scheduler job: `mose-r512-c2-r1-p1-2x8-0930-1746-72843868`
  - launcher: `run_c2_k_silu_down_rms_gateup.sh`
  - auto restart disabled.
- Immediate status at 17:46:49: C2 is `Running`, 2/2 active, no failed/stopped replicas. Startup logs confirm all 16 ranks connected; initialization is still in progress, with no traceback or OOM.

## 2026-09-30 18:20--18:21 HKT — eighth check

All five jobs are `Running`, each with 2/2 active replicas and zero failed/stopped replicas.

- C1: step 8850/14306, train CE 2.657, grad norm 0.0181, ETA about 2h19m.
- C3: step 7650/14306, train CE 2.627, grad norm 0.0243, ETA about 3h00m.
- C4: step 5290/14306, train CE 2.642, grad norm 0.0301, ETA about 3h46m.
- C0: step 2180/14306, train CE 2.870, grad norm 0.0641, ETA about 5h12m.
- C2: step 1190/14306, train CE 3.125, grad norm 0.2531, ETA about 5h48m.

C2 completed initialization and is training normally. No OOM, NaN, traceback, scheduler failure, or resubmission was observed.

## 2026-09-30 18:52--18:53 HKT — ninth check

All five jobs remain `Running`, each with 2/2 active replicas and zero failed/stopped replicas.

- C1: step 10110/14306, train CE 2.661, grad norm 0.0187, ETA about 1h47m.
- C3: step 8860/14306, train CE 2.600, grad norm 0.0678, ETA about 2h27m.
- C4: step 6570/14306, train CE 2.695, grad norm 0.0243, ETA about 3h14m.
- C0: step 3440/14306, train CE 2.749, grad norm 0.0255, ETA about 4h39m.
- C2: step 2420/14306, train CE 2.905, grad norm 0.0354, ETA about 5h14m.

No OOM, NaN, traceback, scheduler failure, or resubmission was observed.

## 2026-09-30 19:24--19:25 HKT — tenth check

All five jobs remain `Running`, each with 2/2 active replicas and zero failed/stopped replicas.

- C1: step 11390/14306, train CE 2.613, grad norm 0.0153, ETA about 1h14m.
- C3: step 10030/14306, train CE 2.594, grad norm 0.0868, ETA about 1h56m.
- C4: step 7870/14306, train CE 2.681, grad norm 0.0224, ETA about 2h41m.
- C0: step 4680/14306, train CE 2.664, grad norm 0.0458, ETA about 4h08m.
- C2: step 3620/14306, train CE 2.858, grad norm 0.0226, ETA about 4h43m.

No OOM, NaN, traceback, scheduler failure, or resubmission was observed.

## 2026-09-30 19:57 HKT — eleventh check

All five jobs remain `Running`, each with 2/2 active replicas and zero failed/stopped replicas.

- C1: step 12610/14306, train CE 2.556, grad norm 0.0199, ETA about 44m.
- C3: step 11230/14306, train CE 2.626, grad norm 0.1409, ETA about 1h23m.
- C4: step 9150/14306, train CE 2.637, grad norm 0.0204, ETA about 2h09m.
- C0: step 5950/14306, train CE 2.574, grad norm 0.0204, ETA about 3h35m.
- C2: step 4850/14306, train CE 2.672, grad norm 0.0238, ETA about 4h10m.

No OOM, NaN, traceback, scheduler failure, or resubmission was observed.

## 2026-09-30 20:29 HKT — twelfth check

All five jobs remain `Running`, each with 2/2 active replicas and zero failed/stopped replicas.

- C1: step 13890/14306, train CE 2.502, grad norm 0.0104, ETA about 11m.
- C3: step 12440/14306, train CE 2.546, grad norm 0.0367, ETA about 51m.
- C4: step 10460/14306, train CE 2.533, grad norm 0.0192, ETA about 1h36m.
- C0: step 7190/14306, train CE 2.634, grad norm 0.0215, ETA about 3h03m.
- C2: step 6060/14306, train CE 2.748, grad norm 0.0225, ETA about 3h38m.

No OOM, NaN, traceback, scheduler failure, or resubmission was observed. C1 is expected to finish before the next 30-minute check.

## 2026-09-30 21:01--21:04 HKT — thirteenth check

- C1 completed successfully: scheduler `Succeeded`, active=0, succeeded=2, failed=0, stopped=0. Logs confirm step 14306/14306, final checkpoint `stage1/step14306`, W&B finalization, and `Training complete`. Its last periodic report at step 14300 had train CE 2.504 and grad norm 0.0073.
- Exact C1 W&B segment:
  `/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20260930_143602-olmo3-1b-r512matched-c1-k-silu-down-rms-kv-fp8-30b`
- Invoked `wandb_sjy` for that segment and verified `Finished syncing`; W&B run is `olmo3-1b-r512matched-c1-k-silu-down-rms-kv-fp8-30b`.
- C3 remains `Running`, 2/2 active, no failed/stopped replicas: step 13660/14306, train CE 2.435, grad norm 0.0299, ETA about 18m.
- C4 remains `Running`, 2/2 active, no failed/stopped replicas: step 11820/14306, train CE 2.572, grad norm 0.0139, ETA about 1h02m.
- C0 remains `Running`, 2/2 active, no failed/stopped replicas: step 8520/14306, train CE 2.578, grad norm 0.0211, ETA about 2h29m.
- C2 remains `Running`, 2/2 active, no failed/stopped replicas: step 7380/14306, train CE 2.571, grad norm 0.0143, ETA about 3h03m.
- No failed job required resubmission.

## 2026-09-30 21:36--21:38 HKT — fourteenth check

- C3 completed successfully: scheduler `Succeeded`, active=0, succeeded=2, failed=0, stopped=0. Logs confirm step 14306/14306, final checkpoint `stage1/step14306`, W&B finalization, and `Training complete`. Its last periodic report at step 14300 had train CE 2.501 and grad norm 0.0074.
- Exact C3 W&B segment:
  `/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20260930_145324-olmo3-1b-r512matched-c3-k-silu-down-rms-kv-gateup-fp8-30b`
- Invoked `wandb_sjy` for that segment and verified `Finished syncing`; W&B run is `olmo3-1b-r512matched-c3-k-silu-down-rms-kv-gateup-fp8-30b`.
- C4 remains `Running`, 2/2 active, no failed/stopped replicas: step 13140/14306, train CE 2.519, grad norm 0.0215, ETA about 30m.
- C0 remains `Running`, 2/2 active, no failed/stopped replicas: step 9860/14306, train CE 2.584, grad norm 0.0303, ETA about 1h54m.
- C2 remains `Running`, 2/2 active, no failed/stopped replicas: step 8660/14306, train CE 2.625, grad norm 0.0205, ETA about 2h29m.
- C1 remains confirmed completed and synced from the prior check. No failed job required resubmission.

## 2026-09-30 22:10--22:12 HKT — fifteenth check

- C4 completed successfully: scheduler `Succeeded`, active=0, succeeded=2, failed=0, stopped=0. Logs confirm step 14306/14306, final checkpoint `stage1/step14306`, W&B finalization, and `Training complete`. Its last periodic report at step 14300 had train CE 2.503 and grad norm 0.0070.
- Exact C4 W&B segment:
  `/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20260930_160839-olmo3-1b-r512matched-c4-attention-silu-down-rms-kv-fp8-30b`
- Invoked `wandb_sjy` for that segment and verified `Finished syncing`; W&B run is `olmo3-1b-r512matched-c4-attention-silu-down-rms-kv-fp8-30b`.
- C0 remains `Running`, 2/2 active, no failed/stopped replicas: step 11180/14306, train CE 2.628, grad norm 0.0198, ETA about 1h21m.
- C2 remains `Running`, 2/2 active, no failed/stopped replicas: step 9980/14306, train CE 2.699, grad norm 0.0219, ETA about 1h54m.
- C1, C3, and C4 are now completed and synced. No failed job required resubmission.

## 2026-09-30 22:44--22:45 HKT — sixteenth check

- C0 remains `Running`, 2/2 active, no failed/stopped replicas: step 12480/14306, train CE 2.596, grad norm 0.0201, ETA about 47m.
- C2 remains `Running`, 2/2 active, no failed/stopped replicas: step 11210/14306, train CE 2.660, grad norm 0.0235, ETA about 1h22m.
- C1, C3, and C4 remain confirmed completed and synced. No failed job required resubmission.

## 2026-09-30 23:16--23:17 HKT — seventeenth check

- C0 remains `Running`, 2/2 active, no failed/stopped replicas: step 13680/14306, train CE 2.429, grad norm 0.0101, ETA about 17m.
- C2 remains `Running`, 2/2 active, no failed/stopped replicas: step 12450/14306, train CE 2.565, grad norm 0.0141, ETA about 49m.
- C1, C3, and C4 remain confirmed completed and synced. No failed job required resubmission.

## 2026-09-30 23:48--23:50 HKT — eighteenth check

- C0 completed successfully: scheduler `Succeeded`, active=0, succeeded=2, failed=0, stopped=0. Logs confirm step 14306/14306, final checkpoint `stage1/step14306`, W&B finalization, and `Training complete`. Its last periodic report at step 14300 had train CE 2.502 and grad norm 0.0072.
- Exact C0 W&B segment:
  `/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20260930_172412-olmo3-1b-r512matched-c0-k-silu-down-rms-fp8-30b`
- Invoked `wandb_sjy` for that segment and verified `Finished syncing`; W&B run is `olmo3-1b-r512matched-c0-k-silu-down-rms-fp8-30b`.
- C2 remains `Running`, 2/2 active, no failed/stopped replicas: step 13670/14306, train CE 2.530, grad norm 0.0137, ETA about 17m.
- C0, C1, C3, and C4 are now completed and synced. No failed job required resubmission.

## 2026-10-01 00:22--00:24 HKT — nineteenth and final check

- C2 completed successfully: scheduler `Succeeded`, active=0, succeeded=2, failed=0, stopped=0. Logs confirm step 14306/14306, final checkpoint `stage1/step14306`, W&B finalization, and `Training complete`. Its last periodic report at step 14300 had train CE 2.499 and grad norm 0.0084.
- Exact C2 W&B segment:
  `/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20260930_174903-olmo3-1b-r512matched-c2-k-silu-down-rms-gateup-fp8-30b`
- Invoked `wandb_sjy` for that segment and verified `Finished syncing`; W&B run is `olmo3-1b-r512matched-c2-k-silu-down-rms-gateup-fp8-30b`.
- All five rank-512-matched combination jobs completed successfully, saved final step-14306 checkpoints, and were synced to W&B. No job failed or required resubmission.

## 2026-10-01 13:15 HKT — C5/C6 follow-up preparation

- Added C5: Q-only SiLU, K/V/O Identity, shared Gate/Up rank 568, Down RMSNorm rank 520.
- Added C6: Q/K SiLU, V/O Identity, shared Gate/Up rank 568, Down RMSNorm rank 520.
- Both use unshared Attention, Q/K/V rank 512 and O rank 496. Each exactly matches the 24,117,248 projection-parameter-per-block rank-512 reference budget.
- Attempted C5 as low-priority arch 2x8 with requested name `mose-r512-c5-r1-p1-2x8-1001-1314`.
- Scheduler rejected the submission before creating a job: the `llmarchitecture_gpu_pool` per-job limit is currently 8 GPUs, while 2x8 requests 16 GPUs. C6 was not submitted because the identical request would be rejected. No live or queued C5/C6 job exists from this attempt.

## 2026-10-01 13:35--13:38 HKT — non-pool submission

- A read-only 2x8 resource prediction against non-pool charged group `llmarchitecture_gpu` returned 2/2 schedulable replicas and seven candidate whole-8-GPU H200 nodes. The prediction created no job and is not a reservation.
- Submitted C5 directly to `ailab-llmarchitecture` with non-pool charged group `llmarchitecture_gpu`, 2x8 GPUs, priority 1:
  `mose-r512-c5-r1-p1-2x8-np-1001-1336-79203407`
- Submitted C6 with the same non-pool resource settings:
  `mose-r512-c6-r1-p1-2x8-np-1001-1337-21978910`
- Both jobs reached `Running` with 2/2 active replicas, zero failed/stopped replicas, and device mesh `(dp_replicate=2, dp_shard=8)`. First optimizer-step evidence is pending the next check.

## 2026-10-01 14:09 HKT — first C5/C6 training check

- C5 remains `Running`, 2/2 active and zero failed/stopped replicas. It reached step 1170/14306 with train CE 3.095, grad norm 0.1943, and ETA about 5h40m.
- C6 remains `Running`, 2/2 active and zero failed/stopped replicas. It reached step 1110/14306 with train CE 3.102, grad norm 0.2094, and ETA about 5h50m.
- Both have passed startup and optimizer-step execution with `(dp_replicate=2, dp_shard=8)`. No OOM, NaN, traceback, or scheduler failure was observed.

## 2026-10-01 15:43 HKT — C5/C6 follow-up check

- Both jobs remain `Running`, each with 2/2 active replicas and zero failed/stopped replicas.
- C5 reached step 4840/14306, train CE 2.683, grad norm 0.0226, ETA about 4h04m.
- C6 reached step 4710/14306, train CE 2.721, grad norm 0.0182, ETA about 4h13m.
- No OOM, NaN, traceback, or scheduler failure was observed. These are noisy single-step values; a matched-step window is needed before deciding whether either beats C2.

## 2026-10-01 16:16 HKT — C5/C6 follow-up check

- Both jobs remain `Running`, each with 2/2 active replicas and zero failed/stopped replicas.
- C5 reached step 6090/14306, train CE 2.664, grad norm 0.0348, ETA about 3h32m.
- C6 reached step 5950/14306, train CE 2.569, grad norm 0.0224, ETA about 3h39m.
- Both have passed the step-5364 checkpoint interval. No OOM, NaN, traceback, or scheduler failure was observed; isolated CE values do not establish a ranking.

## 2026-10-01 16:47 HKT — C5/C6 follow-up check

- Both jobs remain `Running`, each with 2/2 active replicas and zero failed/stopped replicas.
- C5 reached step 7330/14306, train CE 2.581, grad norm 0.0235, ETA about 3h.
- C6 reached step 7140/14306, train CE 2.654, grad norm 0.0184, ETA about 3h09m.
- No OOM, NaN, traceback, or scheduler failure was observed. Matched-step train CE ranking remains pending full-history analysis.

## 2026-10-01 17:19 HKT — C5/C6 follow-up check

- Both jobs remain `Running`, each with 2/2 active replicas and zero failed/stopped replicas.
- C5 reached step 8560/14306, train CE 2.654, grad norm 0.0232, ETA about 2h28m.
- C6 reached step 8370/14306, train CE 2.626, grad norm 0.0187, ETA about 2h36m.
- No OOM, NaN, traceback, or scheduler failure was observed. Matched-step train CE ranking remains pending full-history analysis.

## 2026-10-01 17:51 HKT — C5/C6 follow-up check

- Both jobs remain `Running`, each with 2/2 active replicas and zero failed/stopped replicas.
- C5 reached step 9820/14306, train CE 2.580, grad norm 0.0281, ETA about 1h56m.
- C6 reached step 9580/14306, train CE 2.687, grad norm 0.0181, ETA about 2h04m.
- No OOM, NaN, traceback, or scheduler failure was observed. Matched-step train CE ranking remains pending full-history analysis.

## 2026-10-01 18:23 HKT — C5/C6 follow-up check

- Both jobs remain `Running`, each with 2/2 active replicas and zero failed/stopped replicas.
- C5 reached step 11040/14306, train CE 2.554, grad norm 0.0291, ETA about 1h24m.
- C6 reached step 10800/14306, train CE 2.598, grad norm 0.0167, ETA about 1h32m.
- No OOM, NaN, traceback, or scheduler failure was observed. Final window comparison remains pending.

## 2026-10-01 18:54 HKT — C5/C6 follow-up check

- Both jobs remain `Running`, each with 2/2 active replicas and zero failed/stopped replicas.
- C5 reached step 12270/14306, train CE 2.497, grad norm 0.0121, ETA about 53m.
- C6 reached step 11990/14306, train CE 2.580, grad norm 0.0178, ETA about 1h01m.
- No OOM, NaN, traceback, or scheduler failure was observed. Final window comparison remains pending.

## 2026-10-01 19:25 HKT — C5/C6 follow-up check

- Both jobs remain `Running`, each with 2/2 active replicas and zero failed/stopped replicas.
- C5 reached step 13470/14306, train CE 2.477, grad norm 0.0135, ETA about 22m.
- C6 reached step 13150/14306, train CE 2.549, grad norm 0.0133, ETA about 31m.
- No OOM, NaN, traceback, or scheduler failure was observed. Final window comparison remains pending.

## 2026-10-01 19:58--20:01 HKT — C5/C6 completion and comparison

- C5 `mose-r512-c5-r1-p1-2x8-np-1001-1336-79203407` and C6 `mose-r512-c6-r1-p1-2x8-np-1001-1337-21978910` both reached scheduler `Succeeded`, with active=0, succeeded=2, failed=0, stopped=0.
- Both logs confirm step 14306/14306, final `stage1/step14306` checkpoint, W&B finalization, and `Training complete`.
- C5 exact W&B segment: `/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261001_133843-olmo3-1b-r512matched-c5-q-silu-down-rms-gateup-fp8-30b`.
- C6 exact W&B segment: `/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261001_133931-olmo3-1b-r512matched-c6-qk-silu-down-rms-gateup-fp8-30b`.
- Invoked `wandb_sjy` for each segment serially and verified `Finished syncing` for both.
- Read the three exact offline W&B histories for C2/C5/C6. Each contains steps 1--14306. C6 step 12516 has a logging anomaly: CE=0, PPL=1, Z loss=0 and nonzero grad norm 0.01919. Excluded this step from all three runs for matched comparison.
- Last 1000 valid-step mean train CE: C6 2.474763, C2 2.475668, C5 2.475720. C6 gains 0.000905 CE against prior best C2. Full valid-step mean: C6 2.789407, C5 2.791060, C2 2.791299. The C6 gain is small and based on one seed.

## 2026-10-02 15:27 HKT — C8 submission

- C8: Q/K/V/O and shared Gate/Up inter-factor Identity, Down RMSNorm. Only K activation differs from C2. Ranks Q/K/V=512, O=496, Gate/Up=568, Down=520; exact rank-512-reference projection budget 24,117,248 per block. Standard 30B FP8 recipe retained.
- Python/Bash syntax and projection-budget checks passed.
- Submitted non-pool arch (`ailab-llmarchitecture`, charged group `llmarchitecture_gpu`), priority 1, 2x8. Created job `mose-r512-c8-r1-p1-2x8-np-1002-64279730`; auto restart disabled.
- Launcher `run_c8_linear_down_rms_gateup.sh`; W&B name `olmo3-1b-r512matched-c8-linear-down-rms-gateup-fp8-30b`. User requested monitoring every 30 minutes.

## 2026-10-02 15:51–15:53 HKT — pool scheduling race requested

- User authorized a pool candidate and cancellation of the losing queued candidate once one gets capacity. Original non-pool job remained Inqueue at 15:51:44.
- Added isolated pool launcher with RUN_NAME suffix `-pool`, preserving model/training settings and avoiding shared checkpoint writes.
- `gpu_run arch` now resolves to non-pool `llmarchitecture_gpu`. Its created duplicate `mose-r512-c8-r1-p1-2x8-pool-1002-28757088` was immediately stopped; at 15:52:27 it had active=0, stopped=2, and neither replica was assigned a node. No training/W&B segment was produced.
- Explicit pool charged-group submission (requested name `mose-r512-c8-r2-p1-2x8-pool-1002`) was rejected: pool per-job cap is 8 GPUs, while 2x8 requests 16. No pool job was created.
- Asked whether user permits a 1x8 low-priority pool candidate with unchanged global batch and 30B schedule. Awaiting answer; retain original non-pool job and 30-minute monitoring.

## 2026-10-02 15:58 HKT — pool 1x8 authorized and submitted

- User explicitly accepted 8-GPU pool jobs. Submitted `mose-r512-c8-r3-p1-1x8-pool-1002-84899396`, charged group `llmarchitecture_gpu_pool`, priority 1, one 8-GPU replica.
- Global batch and training schedule unchanged; pool W&B/checkpoints isolated with `-pool` suffix.
- At 16:04 both pool and original non-pool C8 candidates remained Inqueue, with zero active replicas. Keep whichever is first allocated and cancel the other as authorized.

## 2026-10-02 16:49 HKT — 30-minute check

C8 non-pool 2x8, C8 pool 1x8, and dense Q-only/K-only pool 1x8 jobs all remain Inqueue with unassigned replicas and zero active/failed/stopped counts. No resubmission or cancellation; no C8 winner yet.

## 2026-10-02 17:20 HKT — non-pool C8 won allocation

- Original C8 non-pool job is Running, active=2, failed/stopped=0. Training reached step 412/14306; at step410 Train CE=3.657, ETA 5h48m.
- Pool C8 was still Inqueue with no assigned node. Stopped it, confirmed active=0/stopped=1, then deleted `mose-r512-c8-r3-p1-1x8-pool-1002-84899396` as requested. Also deleted the already-stopped accidental non-pool duplicate `mose-r512-c8-r1-p1-2x8-pool-1002-28757088`. Neither ever started training; no W&B sync required.

## 2026-10-02 17:52 HKT

C8 Running, 2/2 active, step1590/14306, Train CE3.035, grad norm0.1110, ETA5h34m. Dense Q-only/K-only remain Inqueue, zero active/failed/stopped replicas. No restart needed.

## 2026-10-02 18:23 HKT

All three experiments are Running with zero failed/stopped replicas. C8 has active=2, step2780/14306, Train CE2.788, grad norm0.0220, ETA5h01m. Dense Q-only has active=1, step340, Train CE3.787, grad norm0.1865, ETA14h32m. Dense K-only has active=1, step300, Train CE4.011, grad norm0.2975, ETA14h43m. The two dense ablations have passed startup and are optimizing normally; current steps are not aligned for quality comparison.

## 2026-10-02 19:24 HKT — monitoring

- C8: Running; {'active': 2, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 5140, 'eta': '3h59m', 'grad': 0.0163, 'ce': 2.659}.
- Q: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 1270, 'eta': '14h6m', 'grad': 0.1107, 'ce': 2.92}.
- K: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 1220, 'eta': '14h18m', 'grad': 0.1089, 'ce': 2.944}.

## 2026-10-02 19:54 HKT — monitoring

- C8: Running; {'active': 2, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 6320, 'eta': '3h28m', 'grad': 0.0243, 'ce': 2.69}.
- Q: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 1730, 'eta': '13h38m', 'grad': 0.0745, 'ce': 2.923}.
- K: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 1680, 'eta': '13h50m', 'grad': 0.0845, 'ce': 2.85}.

## 2026-10-02 20:25 HKT — monitoring

- C8: Running; {'active': 2, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 7490, 'eta': '2h58m', 'grad': 0.0258, 'ce': 2.629}.
- Q: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 2200, 'eta': '13h8m', 'grad': 0.0416, 'ce': 2.818}.
- K: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 2140, 'eta': '13h21m', 'grad': 0.0442, 'ce': 2.788}.

## 2026-10-02 20:55 HKT — monitoring

- C8: Running; {'active': 2, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 8660, 'eta': '2h27m', 'grad': 0.0214, 'ce': 2.626}.
- Q: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 2670, 'eta': '12h37m', 'grad': 0.0253, 'ce': 2.656}.
- K: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 2610, 'eta': '12h50m', 'grad': 0.0447, 'ce': 2.749}.

## 2026-10-02 21:26 HKT — monitoring

- C8: Running; {'active': 2, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 9850, 'eta': '1h56m', 'grad': 0.0206, 'ce': 2.539}.
- Q: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 3140, 'eta': '12h7m', 'grad': 0.0233, 'ce': 2.643}.
- K: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 3070, 'eta': '12h20m', 'grad': 0.0222, 'ce': 2.705}.

## 2026-10-02 21:56 HKT — monitoring

- C8: Running; {'active': 2, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 11010, 'eta': '1h26m', 'grad': 0.015, 'ce': 2.632}.
- Q: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 3610, 'eta': '11h36m', 'grad': 0.0164, 'ce': 2.611}.
- K: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 3530, 'eta': '11h50m', 'grad': 0.0187, 'ce': 2.636}.

## 2026-10-02 22:27 HKT — monitoring

- C8: Running; {'active': 2, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 12200, 'eta': '55m', 'grad': 0.0176, 'ce': 2.627}.
- Q: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 4080, 'eta': '11h6m', 'grad': 0.0291, 'ce': 2.621}.
- K: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 4000, 'eta': '11h19m', 'grad': 0.029, 'ce': 2.639}.

## 2026-10-02 22:58 HKT — monitoring

- C8: Running; {'active': 2, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 13360, 'eta': '25m', 'grad': 0.0108, 'ce': 2.504}.
- Q: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 4550, 'eta': '10h35m', 'grad': 0.0179, 'ce': 2.594}.
- K: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 4490, 'eta': '10h43m', 'grad': 0.0353, 'ce': 2.637}.

## 2026-10-02 23:28 HKT — monitoring

- C8: Succeeded; {'active': 0, 'succeeded': 2, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 14300, 'eta': '1m', 'grad': 0.0071, 'ce': 2.498}.
- Q: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 5020, 'eta': '10h4m', 'grad': 0.0151, 'ce': 2.52}.
- K: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 4950, 'eta': '10h13m', 'grad': 0.0175, 'ce': 2.588}.

C8 completed step14306 at23:22 HKT; W&B exact segment offline-run-20261002_170943-olmo3-1b-r512matched-c8-linear-down-rms-gateup-fp8-30b synced, Finished syncing verified23:28. See C8_RESULTS.md.

## 2026-10-03 00:00 HKT — monitoring

- Q: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 5520, 'eta': '9h32m', 'grad': 0.0354, 'ce': 2.562}.
- K: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 5440, 'eta': '9h41m', 'grad': 0.0177, 'ce': 2.595}.

## 2026-10-03 00:18 HKT — monitoring

- Q: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 5790, 'eta': '9h13m', 'grad': 0.0524, 'ce': 2.476}.
- K: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 5690, 'eta': '9h26m', 'grad': 0.0178, 'ce': 2.568}.

## 2026-10-03 00:47 HKT — monitoring

- Q: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 6250, 'eta': '8h43m', 'grad': 0.0139, 'ce': 2.554}.
- K: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 6150, 'eta': '8h56m', 'grad': 0.0155, 'ce': 2.568}.

## 2026-10-03 01:18 HKT — monitoring

- Q: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 6720, 'eta': '8h13m', 'grad': 0.1266, 'ce': 2.53}.
- K: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 6610, 'eta': '8h26m', 'grad': 0.0277, 'ce': 2.541}.

## 2026-10-03 02:00 HKT — monitoring

- Q: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 7380, 'eta': '7h29m', 'grad': 0.0175, 'ce': 2.45}.
- K: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 7260, 'eta': '7h43m', 'grad': 0.0177, 'ce': 2.468}.

## 2026-10-03 02:31 HKT — monitoring

- Q: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 7850, 'eta': '6h59m', 'grad': 0.0321, 'ce': 2.561}.
- K: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 7730, 'eta': '7h12m', 'grad': 0.0241, 'ce': 2.569}.

## 2026-10-03 03:01 HKT — monitoring

- Q: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 8320, 'eta': '6h28m', 'grad': 0.0484, 'ce': 2.502}.
- K: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 8190, 'eta': '6h42m', 'grad': 0.0188, 'ce': 2.423}.

## 2026-10-03 03:32 HKT — monitoring

- Q: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 8790, 'eta': '5h58m', 'grad': 0.0175, 'ce': 2.475}.
- K: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 8650, 'eta': '6h12m', 'grad': 0.0162, 'ce': 2.488}.

## 2026-10-03 04:02 HKT — monitoring

- Q: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 9260, 'eta': '5h28m', 'grad': 0.0507, 'ce': 2.447}.
- K: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 9120, 'eta': '5h41m', 'grad': 0.0172, 'ce': 2.559}.

## 2026-10-03 04:33 HKT — monitoring

- Q: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 9730, 'eta': '4h57m', 'grad': 0.0233, 'ce': 2.514}.
- K: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 9590, 'eta': '5h10m', 'grad': 0.0194, 'ce': 2.501}.

## 2026-10-03 05:03 HKT — monitoring

- Q: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 10200, 'eta': '4h27m', 'grad': 0.0186, 'ce': 2.424}.
- K: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 10050, 'eta': '4h40m', 'grad': 0.0224, 'ce': 2.477}.

## 2026-10-03 05:34 HKT — monitoring

- Q: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 10670, 'eta': '3h56m', 'grad': 0.0223, 'ce': 2.395}.
- K: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 10510, 'eta': '4h10m', 'grad': 0.0139, 'ce': 2.465}.

## 2026-10-03 06:04 HKT — monitoring

- Q: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 11140, 'eta': '3h26m', 'grad': 0.0137, 'ce': 2.461}.
- K: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 10990, 'eta': '3h38m', 'grad': 0.0312, 'ce': 2.489}.

## 2026-10-03 06:34 HKT — monitoring

- Q: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 11600, 'eta': '2h56m', 'grad': 0.0217, 'ce': 2.518}.
- K: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 11460, 'eta': '3h7m', 'grad': 0.0154, 'ce': 2.448}.

## 2026-10-03 07:05 HKT — monitoring

- Q: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 12070, 'eta': '2h25m', 'grad': 0.0267, 'ce': 2.47}.
- K: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 11920, 'eta': '2h37m', 'grad': 0.0186, 'ce': 2.421}.

## 2026-10-03 07:35 HKT — monitoring

- Q: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 12520, 'eta': '1h56m', 'grad': 0.0175, 'ce': 2.455}.
- K: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 12390, 'eta': '2h6m', 'grad': 0.0175, 'ce': 2.43}.

## 2026-10-03 08:06 HKT — monitoring

- Q: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 12990, 'eta': '1h26m', 'grad': 0.0134, 'ce': 2.379}.
- K: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 12830, 'eta': '1h37m', 'grad': 0.0154, 'ce': 2.438}.

## 2026-10-03 08:36 HKT — monitoring

- Q: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 13480, 'eta': '54m', 'grad': 0.0144, 'ce': 2.402}.
- K: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 13290, 'eta': '1h7m', 'grad': 0.0109, 'ce': 2.418}.

## 2026-10-03 09:06 HKT — monitoring

- Q: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 13950, 'eta': '24m', 'grad': 0.0123, 'ce': 2.399}.
- K: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 13760, 'eta': '36m', 'grad': 0.0086, 'ce': 2.333}.

## 2026-10-03 09:37 HKT — monitoring

- Q: Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 14300, 'eta': '1m', 'grad': 0.0079, 'ce': 2.379}.
- K: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 14220, 'eta': '6m', 'grad': 0.0087, 'ce': 2.381}.

## 2026-10-03 10:09 HKT — monitoring

- Q: Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 14300, 'eta': '1m', 'grad': 0.0079, 'ce': 2.379}.
- K: Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 14300, 'eta': '1m', 'grad': 0.0082, 'ce': 2.38}.

## C9 Q-only SiLU＋Down RMS — 2026-10-09T15:10:38+08:00

已提交arch pool 1×8、priority1；Job ID：`xtuner-mose-c9-q-rms-p1-1x8-1009-r1-mv0mlyc4-1b571`。首次调度核验为Running，active=1。

所有投影独立rank512，参数797,050,880；[设计说明](C9_Q_ONLY_RMS.md)／[实际结构与参数审计](C9_CONFIG_AUDIT_20261009.json)／[当前状态](C9_RUN_STATE.json)。最终checkpoint完成前不提交转换/OLMES；原生/HF数值验证仍not_run。
