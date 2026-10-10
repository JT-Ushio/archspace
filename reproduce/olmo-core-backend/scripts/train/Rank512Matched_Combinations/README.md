# Rank-512-budget-matched combination experiments

These experiments keep the projection-parameter budget of the fully unshared
rank-512 MoSE model. They do not match the much larger Dense parameter count.
The fixed reference contains 24,117,248 projection parameters per Transformer
block and approximately 797,050,880 total model parameters.

## Motivation

The primary metric for this sweep is the mean Train CE over the final 1,000
optimizer steps. In the completed Attention activation sweep, K-only SiLU is
best at 2.477119, followed by Q-only at 2.477917, QK at 2.478748 and all-linear
at 2.480808. In the separate sharing sweep, K/V sharing improves Train CE from
2.496082 to 2.488552 and Gate/Up sharing improves it to 2.489119. These
ingredients have not been evaluated together.

An incomplete FFN-linear/Attention-SiLU run provides an additional train-only
lead: over matched steps 7891--8890 it reaches 2.629824, compared with 2.632624
for K-only and 2.634868 for all-linear. It stopped at step 8890, so it is kept
as a higher-risk candidate rather than treated as a completed 30B winner.

All new configurations therefore use:

- SiLU only between the K factors for C0--C3; Q/V/O remain Identity;
- Identity between the Gate/Up factors for C0--C6 while preserving the native SwiGLU; C7 tests SiLU there;
- RMSNorm without learnable gain between the Down factors;
- the standard 30B FP8 recipe.

C4 instead applies SiLU to Q/K/V/O to follow up the incomplete train-only lead.
C7 uses SiLU on Q/K and Gate/Up while retaining Identity on V/O.

## Exact parameter matching

For d_model=2048 and FFN width=8192, the unshared rank-512 reference is:

    Attention: 4 * (2048*512 + 512*2048) =  8,388,608
    Gate/Up:    2 * (2048*512 + 512*8192) = 10,485,760
    Down:           8192*512 + 512*2048   =  5,242,880
    Total per block:                           24,117,248

Every candidate below matches 24,117,248 exactly. Ranks are multiples of eight
to retain favorable FP8 matrix shapes. Small Q/O/Down adjustments compensate
for divisibility when a shared rank alone cannot exactly match the reference.

| Run | Attention ranks and sharing | Gate/Up | Down | Role |
| --- | --- | --- | --- | --- |
| C0 | K-only SiLU; Q/K/V/O=512, unshared | 512, unshared | 512 | clean Train-CE activation control |
| C1 | K-only SiLU; Q=512, K/V=688 shared, O=504 | 512, unshared | 512 | primary isolated KV candidate |
| C2 | K-only SiLU; Q/K/V=512, O=496, unshared | 568, shared | 520 | isolated Gate/Up candidate |
| C3 | K-only SiLU; Q=520, K/V=680 shared, O=512 | 568, shared | 512 | combined sharing candidate |
| C4 | Q/K/V/O SiLU; Q=512, K/V=688 shared, O=504 | 512, unshared | 512 | higher-risk train-only candidate |
| C5 | Q-only SiLU; Q/K/V=512, O=496, unshared | 568, shared | 520 | Q-SiLU × Gate/Up-sharing interaction |
| C6 | Q/K SiLU; Q/K/V=512, O=496, unshared | 568, shared | 520 | QK-SiLU × Gate/Up-sharing interaction |
| C7 | Q/K SiLU; Q/K/V=512, O=496, unshared | 568, shared, Gate/Up SiLU | 520 | C6 with Gate/Up inter-factor SiLU; Down RMSNorm |

## Recommended run order

1. Run C1 first. It combines the completed 30B Train-CE winner, K-only SiLU,
   with the strongest sharing result, K/V sharing.
2. Run C3 next to test whether the K/V and Gate/Up Train-CE gains compose.
3. Run C4 when a third slot is available. It follows the best matched-step
   Train CE, but its parent run did not complete 30B.
4. C0 and C2 are attribution controls. C0 measures the small Down-RMS change;
   C2 isolates Gate/Up sharing in the K-only background.

Compare all runs at matched optimizer steps and tokens. Use the full W&B
history for the last-1000-step Train CE and the same 11-source validation macro
CE. A 7.5B checkpoint can be used for an early read, but the scheduler must
retain its 30B horizon.

## Fixed recipe

- global batch: 256 sequences / 2,097,152 tokens
- sequence length: 8192
- rank microbatch: 32,768 tokens
- budget: 30B tokens / about 14,306 optimizer steps
- Muon LR: 0.005
- WSD: 2,000-step warmup, final 20% decay
- Float8 conversion enabled
- checkpoint every 1,788 steps; retain all checkpoints

Launchers:

    scripts/train/Rank512Matched_Combinations/run_c0_k_silu_down_rms.sh
    scripts/train/Rank512Matched_Combinations/run_c1_k_silu_down_rms_kv.sh
    scripts/train/Rank512Matched_Combinations/run_c2_k_silu_down_rms_gateup.sh
    scripts/train/Rank512Matched_Combinations/run_c3_k_silu_down_rms_kv_gateup.sh
    scripts/train/Rank512Matched_Combinations/run_c4_attention_silu_down_rms_kv.sh
    scripts/train/Rank512Matched_Combinations/run_c5_q_silu_down_rms_gateup.sh
    scripts/train/Rank512Matched_Combinations/run_c6_qk_silu_down_rms_gateup.sh
    scripts/train/Rank512Matched_Combinations/run_c7_qk_silu_ug_silu_down_rms_gateup.sh

The launchers prepare training commands only; no GPU job is submitted here.

Completed C5/C6 train CE comparison: [C5_C6_RESULTS.md](C5_C6_RESULTS.md).
Completed C0–C7 matched comparison: [C0_C7_RESULTS.md](C0_C7_RESULTS.md).

C8 adds the missing all-linear-projection + shared Gate/Up + Down RMSNorm control. It differs from C2 only in K inter-factor activation (Identity instead of SiLU), with identical ranks and training recipe. Launcher: `run_c8_linear_down_rms_gateup.sh`.
