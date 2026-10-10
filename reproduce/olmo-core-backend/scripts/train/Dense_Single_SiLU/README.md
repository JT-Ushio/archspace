# Single Q/K SiLU at the dense baseline parameter budget

These two runs replace only Q or only K in every OLMo3-1B attention layer with `SiLU(x A) B`. All other attention projections and the entire FFN remain the original dense modules. In particular, these runs do not insert a Down bottleneck/RMSNorm or share Gate/Up. Original QK normalization, RoPE, residual/block norms and SwiGLU remain unchanged.

For the selected 2048-to-2048 projection, rank 1024 gives exactly `2048*1024 + 1024*2048 = 2048*2048` weights. The config checks that the full parameter count equals the dense baseline. The selected factors use existing MoSE initialization (input factor std 0.02; output factor std 1/sqrt(rank)); untouched dense projections retain the normal dense initializer. This evaluates a nonlinear factorization, not an output SiLU applied to a dense matrix.

| Variant | Q | K | V/O/FFN |
| --- | --- | --- | --- |
| Q-only | SiLU(x A_Q) B_Q, rank 1024 | dense | dense |
| K-only | dense | SiLU(x A_K) B_K, rank 1024 | dense |

Recipe: 30B tokens, sequence length 8192, global batch 256 sequences / 2,097,152 tokens, Muon LR 0.005, WSD warmup 2000 and final 20% decay, FP8 enabled. Microbatch is the safe 32,768 tokens/device (4 sequences); with 1x8 GPUs there are eight accumulation microbatches per step. The original dense launcher used 65,536-token microbatches; global batch and schedule are preserved. Checkpoints every 1788 steps, retain all. No new dense baseline is submitted.

User authorized pool 1x8 low-priority jobs on 2026-10-02. Compare matched-step Train CE, primarily the last 1000 steps, with the existing dense baseline; report any zero-CE logging rows consistently.

Launchers: `run_q_only_silu.sh`, `run_k_only_silu.sh`.

## Proposed Up-only / Gate-only extension (2026-10-03)

User authorized running remaining experiments immediately if resources are available. Each proposed extension replaces only one FFN input projection with `SiLU(x A) B`, with independent factors and all attention and Down projections unchanged. The original outer SwiGLU gate SiLU remains, including in the Gate-only variant.

For dimensions 2048 -> 8192, exact same-count factorization requires noninteger rank 1638.4. Prepared launchers use FP8-aligned rank1632, giving 16,711,680 instead of16,777,216 weights per selected projection. Whole model:1,483,868,160 versus dense1,484,916,736 (1,048,576 fewer, 0.070614% lower). These are explicitly named `dense-nearmatched`, not exact matched. User explicitly accepted this deviation on2026-10-03; these remain near-matched, not exact matches.

Launchers: `run_up_only_silu.sh`, `run_gate_only_silu.sh`. Recipe matches Q/K runs. Initialization uses input factor std0.02 and output factor1/sqrt(rank); unchanged dense projections retain normal std0.02. Tests cover actual forward/backward, whole-model and module counts, unchanged attention, and fallback dense initialization.
