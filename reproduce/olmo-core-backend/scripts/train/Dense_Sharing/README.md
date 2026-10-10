# Sharing at the dense parameter budget

User requested dense-budget sharing ablations, ranks rounded downward, 30-minute monitoring, prefer16 GPUs. Preserve prior fallback: if arch2x8 queues, stop it and use pool1x8 at priority1. Do not run duplicate jobs against one output directory.

| Variant | UG | KV | Total parameters | Delta vs dense |
|---|---|---|---:|---:|
| Dense existing | dense independent | dense independent |1484916736|0|
| S1 | shared linear rank1808 | dense independent |1481246720|-0.247153%|
| S2 | dense independent | shared linear rank1360 |1484392448|-0.035308%|
| S3 | shared linear rank1808 | shared linear rank1360 |1480722432|-0.282461%|

UG: z=x A_ug, up=z B_up, gate=z B_gate; output=(SiLU(gate)*up) W_down. KV: z=x A_kv, K=z B_k, V=z B_v. Shared input projection computed once, output projections independent. No added intermediate activations/norms. Original QK normalization, RoPE/sliding-window configuration, residual norms, SwiGLU gate and dense Down retained. Q/O remain original dense in every run. This changes factorization and sharing jointly; it is not a sharing-only causal contrast against dense.

Ranks: floor16(2*2048*8192/(2048+2*8192))=1808; floor16(2*2048*2048/(3*2048))=1360. No padding parameters. New factors use existing MoSE initialization: input std0.02, outputs1/sqrt(rank). Unchanged dense projections retain baseline initialization.

Recipe:30B tokens, seq8192, global batch256/2097152 tokens, Muon0.005, WSD2000-step warmup/final20% decay, FP8 enabled. Rank microbatch32768 tokens,8 accumulation microbatches on1x8 or4 on2x8 (DP size16). Checkpoint interval1788, retain all. Config inherits original dense model/seed recipe. Unique run names `olmo3-1b-dense-sharing-s{1,2,3}-linear-fp8-30b`.

Primary metric: same-step last1000 Train CE; secondary full-training average. Compare existing dense baseline; disclose parameter deltas and factorization initializer. Launcher scripts run_s1.sh,run_s2.sh,run_s3.sh. Local tests verify KV attention matches its collapsed dense matrices, FFN formula, backward, parameter counts and untouched module configs.
