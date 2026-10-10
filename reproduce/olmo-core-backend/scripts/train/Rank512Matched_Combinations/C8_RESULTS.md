# C8 completed comparison

C8 completed all 14306 steps on 2026-10-02 23:22 HKT, scheduler Succeeded with 2 succeeded and 0 active replicas. Final checkpoint save logged at step14306.

Exact W&B segment: `/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261002_170943-olmo3-1b-r512matched-c8-linear-down-rms-gateup-fp8-30b`. Sync completed; `Finished syncing` verified at 23:28 HKT.

All histories contain 14306 Train CE rows. Match earlier C0–C7 report: exclude steps12516 and14304 from every run; full mean14304 points, last1000 window13307–14306 has999 points. C8 has no nonpositive CE rows.

| Run | Full mean Train CE | Last1000 Train CE | Final raw CE | Val macro CE@14000 |
|---|---:|---:|---:|---:|
| C2 | 2.791325 | 2.475732 | 2.505400 | 2.734792 |
| C6 | 2.789434 | 2.474827 | 2.504259 | 2.736803 |
| C8 | 2.790090 | 2.475791 | 2.504908 | 2.737522 |

C8 differs from C2 only by K intermediate activation Identity rather than SiLU. C6 uses Q/K SiLU. All three share Up/Gate, use Identity UG intermediate activation and Down RMSNorm. C8 last-window CE is effectively tied with C2 (+0.000059), slightly worse than C6 (+0.000964). Single-seed results do not establish significance. Standard SwiGLU gate SiLU remains present.
