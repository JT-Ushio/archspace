# Dense-matched Q/K-only SiLU results

Both runs completed14306 steps (30B budget), Q at2026-10-03 09:30 HKT, K09:42 HKT. Scheduler Succeeded, each0 active/1 succeeded. Final14306 checkpoint saves logged. Each model1,484,916,736 parameters, matching dense baseline. Only selected Q or K is factorized as SiLU(xA)B at rank1024; original FFN/Down unchanged. Runs used1x8 pool, priority1.

All runs have14306 Train CE values, no missing/nonpositive/nonfinite rows. Full mean uses steps1–14306; last1000 uses13307–14306. No smoothing. These are single-seed observations, not evidence of statistical significance.

| Run | Full mean Train CE | Last1000 Train CE | Final raw CE |
|---|---:|---:|---:|
| dense | 2.664922596 | 2.357587371 | 2.388383150 |
| q | 2.667828515 | 2.357467091 | 2.387946606 |
| k | 2.665999208 | 2.356372137 | 2.386628866 |

K-only has best tail mean, 0.001215 below dense. Q-only tail effectively ties dense (-0.000120). Full-training means favor dense: Q +0.002906, K +0.001077. No convincing large improvement.

Baseline reconstructed from exact offline segments20260902_120045 through3576 and20260902_162134 after3576, run `olmo3-1b-dense-baseline-bs256-seq8k-30b-wsd-fp8-5e-3`. Recipe30B, batch256/seq8192, Muon.005, WSD2000 warmup/final20% decay, FP8. New jobs use32768-token microbatch vs original baseline65536, and checkpoint interval1788 vs3576; global batch/schedule retained. Changed projections use MoSE factor initialization; untouched dense weights retain baseline initializer.

New W&B segments under `/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb`:
- `offline-run-20261002_180048-olmo3-1b-dense-matched-q-only-silu-r1024-fp8-30b`: sync completed09:37.
- `offline-run-20261002_180311-olmo3-1b-dense-matched-k-only-silu-r1024-fp8-30b`: sync completed, Finished syncing verified10:10 HKT.

## Completed Up/Gate extension — 2026-10-04

User accepted rank1632 near-matched budget. Gate completed07:26:22 HKT, Up07:41:10 HKT; both scheduler Succeeded with0 active/1 succeeded, final14306 checkpoint saves logged. Neither needed a training restart. Each uses1,483,868,160 parameters (0.070614% below dense); Q/K variants match dense exactly. Original SwiGLU gate SiLU and dense Down retained. Only selected input projection has a new intermediate SiLU, no sharing.

All five runs have14306 finite positive CE rows; no exclusions needed.

| Run | Full mean Train CE | Last1000 Train CE | Final raw CE | Tail delta vs dense |
|---|---:|---:|---:|---:|
| dense | 2.664922596 | 2.357587371 | 2.388383150 | +0.000000000 |
| q | 2.667828515 | 2.357467091 | 2.387946606 | -0.000120281 |
| k | 2.665999208 | 2.356372137 | 2.386628866 | -0.001215234 |
| up | 2.672786899 | 2.363875898 | 2.394327164 | +0.006288527 |
| gate | 2.672208942 | 2.361812764 | 2.391754627 | +0.004225393 |

Gate and Up both worsen Train CE versus dense. K remains best tail among these variants, but its gain is only0.001215 and full-training mean still favors dense. No robust overall improvement established.

Exact W&B offline segments under same root:
- Gate: `offline-run-20261003_160237-olmo3-1b-dense-nearmatched-gate-only-silu-r1632-fp8-30b`; Finished syncing verified08:04 HKT.
- Up: `offline-run-20261003_160137-olmo3-1b-dense-nearmatched-up-only-silu-r1632-fp8-30b`; Finished syncing verified08:06 HKT.
