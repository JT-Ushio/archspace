# L/Q/K checkpoint analysis — 2026-09-12

Job `mose-qk-checkpoint-analysis-2x8-0912-29594700` completed on both nodes with
exit code 0. All 16 worker markers and the complete manifest were checked locally.
Analysis source hashes match the executed sources; 10 SVG files parse correctly.

## Design and coverage

L = all-linear; Q = Q-only SiLU; K = K-only SiLU. All checkpoints were trained
with FP8. Evaluate each checkpoint with BF16 and its saved FP8 recipe, on the
same 128 source-balanced held-out sequences, with full 8192-token input context
and label-valid sampled next-token positions. There are 15,070 scored tokens
per condition, 4 checkpoint stages (3576/7152/10728/14306) and 3,072
sequence-level scores across all conditions. Inference batch size is one sequence.

## Final checkpoint: sampled CE (lower is better)

| Model | BF16 forward | FP8 forward | FP8 minus BF16 |
|---|---:|---:|---:|
| L | 2.935756 | 2.935900 | +0.000144 |
| Q | 2.948559 | 2.955609 | +0.007050 |
| K | 2.954991 | 2.956209 | +0.001218 |

These measurements do **not** support assuming Q/K SiLU is better than L on
held-out data. They complement, rather than replace, the full validation
history in [the control summary](CONTROL_RESULTS_20260912.md): Q/K had lower
terminal training loss but not lower final macro validation CE.

Paired sequence-bootstrap 95% intervals for **variant minus L** at the final
checkpoint are:

| Comparison | BF16 forward | FP8 forward |
|---|---|---|
| Q minus L | +0.012803 [-0.004727, +0.031399] | +0.019710 [+0.001073, +0.039131] |
| K minus L | +0.019235 [+0.001528, +0.039914] | +0.020309 [+0.002006, +0.041717] |

The final same-checkpoint FP8-minus-BF16 interval for Q is
[+0.004016, +0.010295]; L and K intervals include zero. Q's forward precision
gap is positive at all four measured stages (about 0.0056–0.0071 CE).
This makes Q's numerical sensitivity worth further investigation. A paired
difference-of-differences calculation is provided below; individual intervals
alone would not establish a between-model precision-sensitivity difference.
Precision also cannot explain away all differences:
Q/K point-estimate CE remains above L under BF16 forward evaluation.

### Paired excess precision penalty relative to L

For each sequence i, calculate
`d_i = (CE_variant_fp8_i - CE_variant_bf16_i) - (CE_L_fp8_i - CE_L_bf16_i)`.
Positive means a greater FP8 forward penalty than L on that sequence. Apply
the same `report.paired_ci` routine (2,000 paired sequence resamples, RNG seed
34521) to the 128 differences, using `per_sequence_ce.csv`. No GPU rerun or
change to the executed analysis sources is involved.

| Checkpoint step | Q excess penalty [95% interval] | K excess penalty [95% interval] |
|---|---|---|
| 3576 | +0.002724 [-0.000088, +0.005847] | -0.000892 [-0.004203, +0.002339] |
| 7152 | +0.003482 [+0.000028, +0.007070] | +0.001873 [-0.002162, +0.006140] |
| 10728 | +0.003784 [+0.000582, +0.006867] | -0.000352 [-0.003407, +0.002664] |
| 14306 | +0.006906 [+0.002277, +0.011913] | +0.001074 [-0.002835, +0.004940] |

This supports a larger forward precision penalty for Q versus L in the final
checkpoint on this sample. K's intervals include zero at every measured stage;
that is not proof of equivalence. These exploratory intervals are unadjusted
for multiple comparisons and do not quantify training-seed uncertainty. The
final Q interval is clearer than the nearly-zero lower endpoint at step7152.

## Attention observations

Q does not simply produce universally sharper attention. At the final
checkpoint with BF16 forward, layer16 normalized entropy is L=0.7259,
Q=0.7698, K=0.7325; maximum attention probability is L=0.2225, Q=0.1798,
K=0.2175. Q is more diffuse in this layer, while earlier layers differ.
These are descriptive within-layer observations, not evidence that higher
or lower entropy causes better prediction.

## Figures and raw provenance

All artifacts are under
[`results/mose-qk-checkpoint-analysis-2x8-0912-29594700`](results/mose-qk-checkpoint-analysis-2x8-0912-29594700).

- [Sampled FP8 CE through training](results/mose-qk-checkpoint-analysis-2x8-0912-29594700/plots/ce-fp8.svg)
- [Same-checkpoint precision gap](results/mose-qk-checkpoint-analysis-2x8-0912-29594700/plots/precision-gap.svg)
- [Layer-wise normalized attention entropy](results/mose-qk-checkpoint-analysis-2x8-0912-29594700/plots/attention-normalized_entropy.svg)
- [Layer-wise maximum attention probability](results/mose-qk-checkpoint-analysis-2x8-0912-29594700/plots/attention-max_probability.svg)
- [Paired effects and intervals](results/mose-qk-checkpoint-analysis-2x8-0912-29594700/paired_effects.csv)

## Boundaries and next decision

The CE estimator is equal-sequence sampled CE, not official full validation
CE or BPB. Intervals reflect sequence-sampling uncertainty conditional on
these trained models; they are not training-seed replication, and multiple
comparisons were not adjusted. FP8/BF16 evaluation changes forward GEMMs only,
does not undo FP8 training, and does not separate quantization from accumulation
error. Batch-one FP8 scaling need not match training microbatches. Attention
probabilities are FP32 reconstructions with actual masks, not bit-exact
FlashAttention internals. Individual cross-model channels are not aligned.

Update 2026-09-13: the Q+K training run is complete. Its matched validation
history and L/Q/K/QK interaction are reported in
[FINAL_RESULTS_20260913.md](FINAL_RESULTS_20260913.md). This checkpoint analysis
still contains only L/Q/K, not QK; no additional GPU analysis was run.
