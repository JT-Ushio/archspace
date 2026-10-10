---
name: mose-experiment-defaults
description: Apply Shijiayang's default hyperparameters when designing or editing MoSE/OLMo-core pretraining experiments in archspace-MoSE, unless the user explicitly overrides them. Includes schedule, comparison conventions and HF export support for architecture changes.
---

# MoSE experiment defaults

Use these defaults for new MoSE/OLMo training experiments for this user unless explicitly overridden:

| Setting | Default |
| --- | --- |
| Global batch | 256 sequences (not per GPU) |
| Sequence length | 8192 tokens |
| Global batch in config | 2,097,152 tokens |
| Training budget | 30,000,000,000 tokens |
| Optimizer | Muon, LR 0.005 |
| Schedule | WSD, 2000 optimizer-step warmup, final 20% decay |
| Precision | Enable Float8 conversion; other operations follow the existing mixed-precision setup |
| Checkpoint cadence | Every 12.5% of the token budget, approximately 1788 optimizer steps |

Follow the user's latest explicit overrides. Do not apply these values retroactively to completed experiments or unrelated code. The earlier Precision_Ablations 7.5B / warmup 500 runs are a deliberate exception, not the default.

Read the existing model and launcher configuration before editing. Match the established WSD class/decay shape unless the user asks for another one; do not silently switch WSD to a sqrt variant. Keep launcher overrides and Python config defaults consistent. Other hyperparameters, architecture, ranks, and sharing choices should follow the experiment's named baseline, not be inferred from this table.

Microbatch is independent of the global batch. The recent memory-safe value was 32768 tokens (4 sequences per device per microbatch); use it for compatible baselines when unspecified, with accumulation to preserve the global batch. Check divisibility against the actual data-parallel size. Do not halve optimizer steps when changing microbatch.

A 30B schedule has roughly 14,306 steps; warmup ends at about 4.194B tokens. For an early 7.5B observation, keep the schedule horizon at 30B. Lowering trainer max_duration may also lower the scheduler horizon; do not implement an early stop this way without verifying/separating those horizons. Existing 7.5B schedules cannot substitute for 30B-schedule control runs at 7.5B.

Periodic checkpoint triggers are approximate due to step rounding. Startup/final saves may add events; a retention count is not a save count. Preserve progress checkpoints (recent config uses max_checkpoints=None), and do not claim disk integrity from W&B save timings alone.

Use distinct experiment directories and unique run names/save paths for new ablations. Keep old experiment results intact. Record resolved configs and seeds. For three-way Down-SiLU/Identity/RMSNorm ablations, use the existing down_nonlinear_u/v branch; Identity returns its input without changing matrices, parameter names or initialization.

For comparisons, align by actual optimizer steps/tokens and match schedules. Standard logs currently expose CE/PPL; BPB requires byte counts of exactly the scored original text and cannot be obtained as CE/log(2) alone. Preserve anomalous rows and report sensitivity analyses explicitly.

When adding a new architecture or changing model forward behavior, read [mose-hf-export](../mose-hf-export/SKILL.md). Every new architecture must include its HF adapter and matching standalone inference support through the unified exporter, plus validation code. New rank/width/activation combinations reuse the existing family module. If the user defers numerical verification, write the checks and explicitly record `not_run`; do not treat a meta audit as numerical validation.
