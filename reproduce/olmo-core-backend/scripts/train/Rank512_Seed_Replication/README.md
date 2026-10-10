# C0/C2/C6 paired initialization-seed replications

The original single-seed final-window Train CE difference is only0.004215
(C0:2.479042; C6:2.474827), so seed confirmation matters for the paper.
These runs change initialization seed only and hold data order fixed.

| Config | Q | K | V/O | Up/Gate | Down | Init seed | Data seed |
|---|---|---|---|---|---|---|---|
| C0s2 | Identity512 | SiLU512 | Identity512 | independent512, Identity | RMSNorm512 |12537|34521|
| C6s2 | SiLU512 | SiLU512 | V Identity512/O Identity496 | shared568, Identity | RMSNorm520 |12537|34521|
| C0s3 | same as C0s2 | | | | |12538|34521|
| C6s3 | same as C6s2 | | | | |12538|34521|
| C2s2 | Identity512 | SiLU512 | V Identity512/O Identity496 | shared568, Identity | RMSNorm520 |12537|34521|
| C2s3 | same as C2s2 | | | | |12538|34521|

C2s2/C2s3 are additional paired controls. They separate
the C2-minus-C0 shared-UG/rank-allocation recipe effect from the much smaller
C6-minus-C2 Q-SiLU effect. C2 and C6 differ only in Q inter-factor activation;
C2 versus C0 also reallocates O/Down/UG ranks, so it is not a pure weight-sharing
comparison. Both were submitted on available pool1x8, priority1, at23:38/23:40
on2026-10-06, matching the existing seed replications. At23:43 each has one
active replica, mesh(1,8), fresh initialization and a saved step0 checkpoint.
C2s2 has a finite first train CE; C2s3's first CE was not yet present in that
startup snapshot and remains to be checked at the next poll.
At23:56 both are training normally, steps270/240 with finite CE; separate
progress_c2s2.log/progress_c2s3.log retain that proof without replacing the
initial mesh/checkpoint evidence.
The completed-cohort exporter reports all three paired contrasts
using one common anomaly mask and rejects unmatched seeds, data order,
allocations or parameter counts.

All configs have797,050,880 parameters, matching the independent rank512
reference. Native SwiGLU remains intact. Full30B, sequence8192/global256,
microbatch32768 tokens, Muon0.005, WSD warmup2000/final20%decay, FP8,
checkpoint1788 steps with all retained. Distinct run/output paths isolate seeds.
The launcher captures overrides before preparation helpers, uses job-local
compiler caches, and checks checkpoint storage inside the GPU allocation.

Use full-precision Train CE at matched steps, with the final1000-step window
as the primary measure. Preserve raw histories and report sensitivity to any
anomalous steps. Compare the C6-minus-C0 difference within each seed, then
report per-config mean and dispersion across the original seed12536 and
the two replications. Do not infer a reliable gain from one final minibatch
or treat adjacent training steps as independent seed replicates.

The new cohort exporter reports the two new matched seed pairs first. The
original seed12536 must be recomputed with the same anomaly mask before
combining it into a three-seed estimate. Allocation differences and resumed
RNG state are retained in provenance; original-run summaries are kept as
context until that common-history recomputation is available.
