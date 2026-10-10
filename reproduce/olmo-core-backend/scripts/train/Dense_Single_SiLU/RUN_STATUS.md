# Dense single Q/K SiLU run status

## 2026-10-02 16:17–16:18 HKT — submission

- User requested separate Q-only and K-only SiLU experiments at exactly the original dense parameter budget, with other modules unchanged, and authorized pool 8-GPU jobs.
- Three local CPU tests passed: Q and K forward/backward, exact attention parameter count and selected-module topology, and full model config equivalence outside the selected projection. Full model count for both variants and dense baseline is 1,484,916,736. Bash syntax checks passed. Local shared-storage Python inspection needed a test-process-only frame lookup/cache optimization to avoid very slow filesystem metadata lookups; model code and training environment were not patched for this.
- Both submitted to namespace `ailab-llmarchitecture`, charged group `llmarchitecture_gpu_pool`, priority 1, 1x8 GPUs, auto restart disabled.

| Variant | Scheduler job | W&B name |
| --- | --- | --- |
| Q-only | `mose-dense-q-silu-r1024-r1-p1-1x8-pool-1002-d60f3` | `olmo3-1b-dense-matched-q-only-silu-r1024-fp8-30b` |
| K-only | `mose-dense-k-silu-r1024-r1-p1-1x8-pool-1002-d135a` | `olmo3-1b-dense-matched-k-only-silu-r1024-fp8-30b` |

At 16:18 both were Inqueue, one unassigned replica each and zero active/failed/stopped counts. Monitor with the C8 candidates at 30-minute intervals; queued tasks must not be duplicated. Preserve the full 30B horizon and global batch when using 8 GPUs.

## 2026-10-02 16:49 HKT — 30-minute check

C8 non-pool 2x8, C8 pool 1x8, and dense Q-only/K-only pool 1x8 jobs all remain Inqueue with unassigned replicas and zero active/failed/stopped counts. No resubmission or cancellation; no C8 winner yet.

## 2026-10-02 17:20 HKT

Q-only and K-only pool jobs remain Inqueue with zero active/failed/stopped replicas. C8 non-pool obtained its two nodes; its losing pool candidate was cancelled and deleted.

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

## 2026-10-03 01:31 HKT — Up/Gate preparation and resource prediction

User authorized submitting remaining experiments when resources are available, without waiting for Q/K completion. Prepared separate Up-only/Gate-only nonshared factorization configs/launchers; rank1632 near-match, user clarification on exact-budget deviation pending. Local tests running. `rjob submit --predict-only=true` for pool1x8 priority1 returned no schedulable nodes; created rjob_name empty, no actual job created. New experiments not submitted. Existing Q/K remain running; continue30min monitoring and resource checks.

Local validation completed: all3 unittest cases passed (15.251s test runtime after slow shared-environment import), Up/Gate total1,483,868,160 vs dense1,484,916,736. Launcher Bash syntax passed. Await resource availability and parameter-tolerance clarification before submission.

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

## 2026-10-03 09:37 HKT — Q complete

Q job succeeded active0/succeeded1, training complete09:30:14 and final14306 checkpoint logged. Exact W&B segment `offline-run-20261002_180048-olmo3-1b-dense-matched-q-only-silu-r1024-fp8-30b` under known taoji MoSE W&B root synchronized; Finished syncing verified. Full14306 rows/no zero CE. Full mean2.667828515, last1000 mean2.357467091, final raw2.387946606. Dense14306-row control reconstructed segments20260902_120045 (through3576) and20260902_162134 (3577 onward): full mean2.664922596,last1000 mean2.357587371,final2.388383150. K still running14220. Up/Gate not submitted: tolerance question unanswered and capacity prediction inconclusive (zero total replicas); earlier inference of no capacity was corrected to user.

## 2026-10-03 10:09 HKT — monitoring

- Q: Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 14300, 'eta': '1m', 'grad': 0.0079, 'ce': 2.379}.
- K: Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 14300, 'eta': '1m', 'grad': 0.0082, 'ce': 2.38}.

## 2026-10-03 10:10 HKT — Both Q/K complete and synchronized

K completed14306 at09:42, Succeeded active0/succeeded1; final checkpoint logged. Exact segment offline-run-20261002_180311-olmo3-1b-dense-matched-k-only-silu-r1024-fp8-30b synced, Finished syncing verified. All14306 rows valid. Full mean2.665999208,tail1000 mean2.356372137,final2.386628866. See RESULTS.md for dense comparison. No running experiments remain in this monitoring set. Up/Gate scripts ready/tests passed, awaiting parameter-tolerance clarification; not submitted.

## 2026-10-03 12:01 HKT — Up/Gate submitted

User explicitly accepted rank1632 near-matched budget. Initially requested arch2x8 but then stipulated no arch queue. Up pool r1 `mose-dense-up-silu-r1632-r1-p1-1x8-pool-1003-ea950` stopped before node assignment, active0/stopped1. Arch r2 `mose-dense-up-silu-r1632-r2-p1-2x8-1003-4340788` was Inqueue with unassigned replicas, stopped12:00:47, active0/stopped2. Neither trained or produced W&B segment, so no sync needed.

Active submissions: pool1x8 each, priority1, namespace ailab-llmarchitecture, charged group llmarchitecture_gpu_pool, auto restart disabled:
- Up: `mose-dense-up-silu-r1632-r3-p1-1x8-pool-1003-597fb`; RUN_NAME `olmo3-1b-dense-nearmatched-up-only-silu-r1632-fp8-30b`.
- Gate: `mose-dense-gate-silu-r1632-r1-p1-1x8-pool-10-f3534`; RUN_NAME `olmo3-1b-dense-nearmatched-gate-only-silu-r1632-fp8-30b`.

No sharing, selected intermediate SiLU, all attention and dense Down unchanged. Both scripts tested3 cases successfully. Continue30min checks.

## 2026-10-03 12:01 HKT — monitoring

- UP: Inqueue; {'active': 0, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics None.
- GATE: Inqueue; {'active': 0, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics None.

## 2026-10-03 12:54 HKT — monitoring

- UP: Inqueue; {'active': 0, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics None.
- GATE: Inqueue; {'active': 0, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics None.

## 2026-10-03 13:24 HKT — monitoring

- UP: Inqueue; {'active': 0, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics None.
- GATE: Inqueue; {'active': 0, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics None.

## 2026-10-03 13:54 HKT — monitoring

- UP: Inqueue; {'active': 0, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics None.
- GATE: Inqueue; {'active': 0, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics None.

## 2026-10-03 14:25 HKT — monitoring

- UP: Inqueue; {'active': 0, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics None.
- GATE: Inqueue; {'active': 0, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics None.

## 2026-10-03 14:55 HKT — monitoring

- UP: Inqueue; {'active': 0, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics None.
- GATE: Inqueue; {'active': 0, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics None.

## 2026-10-03 15:25 HKT — monitoring

- UP: Inqueue; {'active': 0, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics None.
- GATE: Inqueue; {'active': 0, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics None.

## 2026-10-03 15:56 HKT — monitoring

- UP: Inqueue; {'active': 0, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics None.
- GATE: Inqueue; {'active': 0, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics None.

## 2026-10-03 16:26 HKT — monitoring

- UP: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 360, 'eta': '14h42m', 'grad': 0.201, 'ce': 3.684}.
- GATE: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 360, 'eta': '14h32m', 'grad': 0.1789, 'ce': 3.686}.

## 2026-10-03 16:57 HKT — monitoring

- UP: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 830, 'eta': '14h39m', 'grad': 0.1079, 'ce': 3.062}.
- GATE: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 830, 'eta': '14h29m', 'grad': 0.1401, 'ce': 3.073}.

## 2026-10-03 17:27 HKT — monitoring

- UP: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 1290, 'eta': '14h14m', 'grad': 0.0991, 'ce': 2.939}.
- GATE: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 1290, 'eta': '14h4m', 'grad': 0.0988, 'ce': 2.94}.

## 2026-10-03 17:57 HKT — monitoring

- UP: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 1740, 'eta': '13h47m', 'grad': 0.0916, 'ce': 2.82}.
- GATE: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 1760, 'eta': '13h34m', 'grad': 0.0689, 'ce': 2.844}.

## 2026-10-03 18:27 HKT — monitoring

- UP: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 2200, 'eta': '13h17m', 'grad': 0.0314, 'ce': 2.824}.
- GATE: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 2230, 'eta': '13h4m', 'grad': 0.0466, 'ce': 2.783}.

## 2026-10-03 18:58 HKT — monitoring

- UP: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 2670, 'eta': '12h46m', 'grad': 0.021, 'ce': 2.663}.
- GATE: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 2700, 'eta': '12h33m', 'grad': 0.0219, 'ce': 2.735}.

## 2026-10-03 19:28 HKT — monitoring

- UP: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 3130, 'eta': '12h16m', 'grad': 0.0345, 'ce': 2.741}.
- GATE: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 3170, 'eta': '12h2m', 'grad': 0.04, 'ce': 2.677}.

## 2026-10-03 19:59 HKT — monitoring

- UP: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 3590, 'eta': '11h46m', 'grad': 0.0563, 'ce': 2.637}.
- GATE: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 3640, 'eta': '11h31m', 'grad': 0.0387, 'ce': 2.643}.

## 2026-10-03 20:29 HKT — monitoring

- UP: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 4050, 'eta': '11h15m', 'grad': 0.0817, 'ce': 2.625}.
- GATE: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 4110, 'eta': '11h1m', 'grad': 0.0293, 'ce': 2.577}.

## 2026-10-03 20:59 HKT — monitoring

- UP: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 4510, 'eta': '10h45m', 'grad': 0.0207, 'ce': 2.586}.
- GATE: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 4580, 'eta': '10h30m', 'grad': 0.0235, 'ce': 2.565}.

## 2026-10-03 21:30 HKT — monitoring

- UP: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 4990, 'eta': '10h10m', 'grad': 0.0374, 'ce': 2.626}.
- GATE: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 5050, 'eta': '9h59m', 'grad': 0.0274, 'ce': 2.553}.

## 2026-10-03 22:00 HKT — monitoring

- UP: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 5460, 'eta': '9h40m', 'grad': 0.0147, 'ce': 2.547}.
- GATE: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 5520, 'eta': '9h29m', 'grad': 0.0195, 'ce': 2.569}.

## 2026-10-03 22:31 HKT — monitoring

- UP: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 5930, 'eta': '9h9m', 'grad': 0.0217, 'ce': 2.529}.
- GATE: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 5990, 'eta': '8h56m', 'grad': 0.0214, 'ce': 2.558}.

## 2026-10-03 23:01 HKT — monitoring

- UP: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 6390, 'eta': '8h39m', 'grad': 0.0443, 'ce': 2.58}.
- GATE: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 6490, 'eta': '8h24m', 'grad': 0.0387, 'ce': 2.497}.

## 2026-10-03 23:31 HKT — monitoring

- UP: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 6850, 'eta': '8h9m', 'grad': 0.0185, 'ce': 2.567}.
- GATE: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 6960, 'eta': '7h54m', 'grad': 0.0217, 'ce': 2.497}.

## 2026-10-04 00:02 HKT — monitoring

- UP: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 7310, 'eta': '7h39m', 'grad': 0.0169, 'ce': 2.5}.
- GATE: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 7430, 'eta': '7h24m', 'grad': 0.0196, 'ce': 2.503}.

## 2026-10-04 00:32 HKT — monitoring

- UP: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 7780, 'eta': '7h8m', 'grad': 0.0242, 'ce': 2.553}.
- GATE: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 7900, 'eta': '6h53m', 'grad': 0.019, 'ce': 2.498}.

## 2026-10-04 01:03 HKT — monitoring

- UP: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 8240, 'eta': '6h38m', 'grad': 0.0283, 'ce': 2.43}.
- GATE: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 8370, 'eta': '6h23m', 'grad': 0.0149, 'ce': 2.51}.

## 2026-10-04 01:33 HKT — monitoring

- UP: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 8710, 'eta': '6h7m', 'grad': 0.019, 'ce': 2.529}.
- GATE: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 8840, 'eta': '5h53m', 'grad': 0.0209, 'ce': 2.493}.

## 2026-10-04 02:04 HKT — monitoring

- UP: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 9170, 'eta': '5h37m', 'grad': 0.0292, 'ce': 2.487}.
- GATE: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 9310, 'eta': '5h23m', 'grad': 0.018, 'ce': 2.543}.

## 2026-10-04 02:34 HKT — monitoring

- UP: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 9620, 'eta': '5h8m', 'grad': 0.0147, 'ce': 2.473}.
- GATE: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 9770, 'eta': '4h53m', 'grad': 0.0197, 'ce': 2.515}.

## 2026-10-04 03:04 HKT — monitoring

- UP: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 10090, 'eta': '4h37m', 'grad': 0.02, 'ce': 2.494}.
- GATE: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 10260, 'eta': '4h21m', 'grad': 0.0141, 'ce': 2.518}.

## 2026-10-04 03:35 HKT — monitoring

- UP: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 10560, 'eta': '4h6m', 'grad': 0.0204, 'ce': 2.457}.
- GATE: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 10730, 'eta': '3h51m', 'grad': 0.0188, 'ce': 2.528}.

## 2026-10-04 04:05 HKT — monitoring

- UP: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 11020, 'eta': '3h36m', 'grad': 0.0142, 'ce': 2.511}.
- GATE: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 11200, 'eta': '3h21m', 'grad': 0.0184, 'ce': 2.488}.

## 2026-10-04 04:36 HKT — monitoring

- UP: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 11490, 'eta': '3h5m', 'grad': 0.0267, 'ce': 2.471}.
- GATE: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 11670, 'eta': '2h50m', 'grad': 0.0207, 'ce': 2.548}.

## 2026-10-04 05:06 HKT — monitoring

- UP: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 11970, 'eta': '2h33m', 'grad': 0.0162, 'ce': 2.443}.
- GATE: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 12140, 'eta': '2h20m', 'grad': 0.0146, 'ce': 2.428}.

## 2026-10-04 05:36 HKT — monitoring

- UP: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 12430, 'eta': '2h3m', 'grad': 0.0166, 'ce': 2.454}.
- GATE: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 12600, 'eta': '1h51m', 'grad': 0.0192, 'ce': 2.372}.

## 2026-10-04 06:07 HKT — monitoring

- UP: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 12870, 'eta': '1h35m', 'grad': 0.0106, 'ce': 2.439}.
- GATE: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 13060, 'eta': '1h21m', 'grad': 0.0124, 'ce': 2.378}.

## 2026-10-04 06:37 HKT — monitoring

- UP: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 13340, 'eta': '1h4m', 'grad': 0.0124, 'ce': 2.366}.
- GATE: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 13530, 'eta': '51m', 'grad': 0.0147, 'ce': 2.375}.

## 2026-10-04 07:07 HKT — monitoring

- UP: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 13790, 'eta': '34m', 'grad': 0.0116, 'ce': 2.384}.
- GATE: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 14000, 'eta': '20m', 'grad': 0.0096, 'ce': 2.426}.

## 2026-10-04 07:38 HKT — monitoring

- UP: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 14260, 'eta': '3m', 'grad': 0.008, 'ce': 2.328}.
- GATE: Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 14300, 'eta': '1m', 'grad': 0.008, 'ce': 2.384}.

## 2026-10-04 08:09 HKT — monitoring

- UP: Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 14300, 'eta': '1m', 'grad': 0.0077, 'ce': 2.386}.
- GATE: Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; latest periodic metrics {'step': 14300, 'eta': '1m', 'grad': 0.008, 'ce': 2.384}.

## 2026-10-04 08:05 HKT — Training finished

Up and Gate both Succeeded active0/succeeded1, completed14306 with final checkpoint logs. See RESULTS.md and train_ce_comparison.json for full-history comparisons. Gate W&B Finished syncing verified; Up exact segment sync launched, verification pending. No live training jobs remain in this monitoring set.

08:06 HKT: Up exact segment Finished syncing verified. All requested Q/K/Up/Gate experiments complete and synchronized. Half-hour monitoring concluded with both Up/Gate terminal-success.
