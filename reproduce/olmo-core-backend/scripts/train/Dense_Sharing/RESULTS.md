# Dense-budget sharing: completed results

All three pool1x8 jobs completed14306 steps on2026-10-06, S1 04:54:59, S2 04:51:20, S3 04:59:06 HKT. Scheduler verified Succeeded active0/succeeded1 for each at10:42; final14306 checkpoint saves recorded in each log. No restart during training. Monitoring gap: last check2026-10-05 17:18, next2026-10-06 10:42 after the waiting tool was aborted; no half-hour checks are claimed during that gap.

Full offline histories for all variants and dense contain14306 valid positive finite CE rows. No exclusions. Full mean steps1–14306; tail mean steps13307–14306,1000 rows. No smoothing.

| Model | Full mean Train CE | Last1000 Train CE | Final raw CE | Tail delta vs dense |
|---|---:|---:|---:|---:|
| Dense | 2.664922596 | 2.357587371 | 2.388383150 | +0.000000000 |
| S1 | 2.683512985 | 2.368455054 | 2.397608995 | +0.010867682 |
| S2 | 2.669288971 | 2.358964991 | 2.388871193 | +0.001377619 |
| S3 | 2.687680738 | 2.371111126 | 2.399370193 | +0.013523755 |

S2(KV sharing) is closest to dense, but none beats dense on either full or last1000 mean. S1(UG sharing) and S3(UG+KV) are worse by0.010868 and0.013524 tail CE. Single seed; the0.001378 S2 gap does not establish significance.

Configuration: linear shared bottlenecks, UG1808/KV1360; Q/O and Down original dense, original SwiGLU gate/QK norms/RoPE retained. Parameters S1=1481246720(-0.247153%),S2=1484392448(-0.035308%),S3=1480722432(-0.282461%),dense=1484916736. Shared factor initialization differs from dense and the changes jointly test sharing/factorization, not isolated sharing. Global batch256,seq8192,30B,Muon.005,WSD2000 warmup/final20% decay,FP8. Safe microbatch32768 versus old dense65536; global batch maintained. See README.md for details.

Baseline exact segments20260902_120045 through3576 and20260902_162134 thereafter, same dense W&B run. Analysis script and summary persisted alongside this report.

## Exact W&B segments

- S1: `/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261005_130419-olmo3-1b-dense-sharing-s1-linear-fp8-30b`
- S2: `/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261005_130632-olmo3-1b-dense-sharing-s2-linear-fp8-30b`
- S3: `/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20261005_130819-olmo3-1b-dense-sharing-s3-linear-fp8-30b`
- dense1: `/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20260902_120045-olmo3-1b-dense-baseline-bs256-seq8k-30b-wsd-fp8-5e-3`
- dense2: `/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/offline-run-20260902_162134-olmo3-1b-dense-baseline-bs256-seq8k-30b-wsd-fp8-5e-3`

Sync status: S1/S2/S3 all Finished syncing verified on2026-10-06.

## Validation at step14000

Same11-source equal-weight macro CE from full offline histories, no smoothing.

| Model | Validation macro CE | Delta vs dense |
|---|---:|---:|
| Dense | 2.626223467 | +0.000000000 |
| S1 | 2.636076136 | +0.009852669 |
| S2 | 2.626938408 | +0.000714941 |
| S3 | 2.637110396 | +0.010886929 |

S2 remains closest; none beats dense on this validation measure. The generalized analysis independently reproduces every previously reported full/tail/final Train CE to1e-12. The dense segments overlap at3577–7050; the later resumed segment takes precedence, matching the existing explicit3576 boundary. Full source list in validation_comparison.json.
