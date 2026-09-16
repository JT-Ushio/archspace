# Q/K nonlinear matching: next experiment

Complete the existing matched 2x2 with Q+K SiLU. Reuse all-linear, Q-only and
K-only; do not train QK+V/QK+O until the QK result supports extending the sweep.
Only Q/K low-rank intermediate activations change; V/O/Gate/Up/Down remain
Identity. Primary SwiGLU, attention softmax, QK norms and outer norms remain.

Rank512, no sharing, FP8, Muon LR0.005, global batch256, sequence8192, 30B,
WSD warmup2000/final20% decay, checkpoints every1788 steps and final14306.
No other baseline hyperparameters change. arch2x8 uses microbatch4 sequences
and four accumulation steps. Existing runs/checkpoints are never overwritten.

Launcher: `run_qk_silu_fp8.sh`; run ID `olmo3-1b-all-linear-qk-silu-fp8-30b`.
Configuration: `cfps/BS256_Seq8K_30B_WSD_FP8/All_Linear_QK_Combination/qk_silu_fp8.py`.

Compare validation CE at aligned optimizer steps/tokens, plus individual-domain
metrics and downstream evaluations. Deduplicate replayed steps after restarts.
Let GQ=L0-LQ, GK=L0-LK, GQK=L0-LQK. The interaction on this loss scale is
I=GQK-GQ-GK. Negative I does not imply QK is worse than Q or K individually.
One seed is descriptive, not proof of seed-robust superiority.

In parallel, analyze trusted saved baseline/Q/K checkpoints on identical held-out
tokens: post-norm/post-RoPE attention, sampled token CE, and BF16/FP8 inference
sensitivity. No training or optimizer loading in the analysis job. Attention
entropy is descriptive, not a monotonic quality metric or causal proof.
