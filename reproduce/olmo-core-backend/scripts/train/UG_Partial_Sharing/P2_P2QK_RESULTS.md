# P2/P2qk full30B Train CE

Both histories contain exactly one usable CE record at every optimizer step1–14306 after chronological resume merging. The common anomaly mask is empty. Native SwiGLU, Down RMSNorm,797,050,880 parameters, data seed34521 and the30B training recipe are shared; P2qk changes only Q/K inter-factor Identity to SiLU.

| Config | Full mean Train CE | Final1000 mean Train CE | Final14306 raw CE | Validation macro CE at14000 |
|---|---:|---:|---:|---:|
|P2|2.798068362|2.484496541|2.513828516|2.746519273|
|P2QK|2.795363770|2.481272976|2.510764122|2.741835692|

Q/K SiLU changes final1000 mean Train CE by-0.003223565 (lower is better). Validation uses the same11 sources at the same step, as an equally weighted source macro. This is a single initialization per architecture; no seed-robust improvement is established.

P2 ran initially on32 GPUs and resumed from step7152 on16 GPUs. Checkpoint model/optimizer/trainer state loaded, but the runtime explicitly warned that rank RNG state was not restored after the world-size change. Its unintended fresh-start r2 segment is excluded. P2qk trained continuously on32 GPUs. Global batch, microbatch and schedule stayed fixed; allocation and RNG provenance remain material limitations of this pair.

Historical C6 final-window CE was approximately2.474827 with a different two-step common anomaly mask. It is a reference value until the original history is recomputed with the new cohort mask. The currentC0/C6 seed replications use matched pool8 allocations and fixed data order.

Authoritative inputs and evidence: P2_P2QK_FULL_TRAIN_CE.json, completed_pair_segments.json, completion_p2.log, completion_p2qk.log, synchronization.json, and reports/p2pair_completion.log. The report job Succeeded/active0; stdout JSON was recovered and its SHA256 matched the emitted digest.
