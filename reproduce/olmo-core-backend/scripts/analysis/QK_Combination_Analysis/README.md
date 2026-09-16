# Q/K matching analysis

Completed 30B QK run and six-way comparison:
[FINAL_RESULTS_20260913.md](FINAL_RESULTS_20260913.md).
`final_compare.py` takes the control snapshot, final QK snapshot and a fresh
output directory, plus explicit `--restored-step` and timezone-aware
`--attempt-started`. It preserves source copies/hashes, rejects inconsistent
history, audits successful-attempt timestamps, and generates matched validation,
training-window, interaction and downstream tables. The observed final API
snapshot has unique optimizer steps; no replay averaging or renumbering is used.

Analyze existing all-linear (L), Q-only (Q), K-only (K) checkpoints at exact
steps3576/7152/10728/14306. Approximately25/50/75/100% training progress;
do not silently use another checkpoint when one is missing/incomplete.

`scripts/train/QK_Combination_Analysis/run_qk_checkpoint_analysis.sh` launches
2x8 independent GPU workers with torchrun. Rank0 prepares128 source-balanced
held-out sequences; each GPU owns8 disjoint sequences, but evaluates all models
and checkpoint stages. Full8K context, up to128 fixed scored positions/sequence,
up to32 query positions for attention statistics, all16layers/16heads.
No optimizer loading, training, checkpoint writes, W&B writes or tokenizer download.
Uses the existing trusted DCP model-loading checks, including all keys/shapes and
finite tensors. Python checkpoint metadata is trusted input, not untrusted pickle.

For each checkpoint, evaluate BF16, then convert that same model in-place with
its saved Float8 recipe and the existing model.apply_fp8 method. LM head stays
BF16 as in training. Record every converted Float8Linear module/config. No
compile/FSDP is used in analysis; this is paired forward sensitivity, not a
bit-exact reproduction of training kernels or a backward precision experiment.
Inference uses one sequence per forward, not training's four-sequence microbatch;
FP8 per-tensor dynamic scales can therefore differ from those used in training.

Attention hooks see Q/K after their norms and RoPE. Reconstruct only sampled
query rows in FP32 with TF32 disabled, matching each backend's scale and window.
Reject packed docs/cache/GQA assumptions rather than applying an incorrect mask.
Save entropy, eligible-key-normalized entropy, logits mean/std/max, maximum
probability, expected distance and disjoint distance-band masses per head/sample.
Snapshots (first2 sequences, layers1/4/8/12/16, final BF16) contain the first4
heads' probability mass in64 absolute-position bins; not channel feature maps.

Next-token CE respects saved validation label masks and right-padding, and scores
the same sampled positions in every model/precision. Report equal-sequence mean
CE and paired sequence bootstrap intervals. Not official full-validation CE,
BPB, training-seed uncertainty, or proof that sharper attention is better.

Outputs go to a new personal-repo directory named from the shared scheduler
JOB_ID. Never use per-node timestamps or overwrite an existing analysis.
Rank0 writes manifest status=complete only after every worker's completion marker
and successful report generation. Failed workers leave explicit error records.
`report.md`, `summary.csv`, `per_sequence_ce.csv`, `paired_effects.csv`,
`attention_summary.csv`,10 SVG plots, raw token/head CSVs and small NPZ snapshots.
The raw sample bundle stores token IDs/masks/positions, not decoded text.

Later report regeneration is CPU-only:

```bash
python scripts/analysis/QK_Combination_Analysis/report.py /path/to/completed/results
```

Training comparison should separately use complete W&B histories aligned by
optimizer steps/tokens with checkpoint-replay duplicates handled explicitly.
These sample statistics complement, not replace, that primary comparison.

## Full-history control comparison

`control_trends.py` reads exported `LABEL_metadata.json`/`LABEL_history.jsonl`
snapshots for L/Q/K/V/O (and optionally QK). It verifies finished states, equal
endpoints, exported config differences, unique steps and logged token counts.
Duplicate/replayed steps fail rather than being silently averaged or dropped.
It compares only complete common validation points and identical final100/500
optimizer-step training windows. It creates a new `trends/` directory and refuses
to overwrite it. No W&B access or GPU is needed to regenerate this comparison.

```bash
python scripts/analysis/QK_Combination_Analysis/control_trends.py SNAPSHOT_DIR --labels L Q K V O
```

Snapshot `results/controls-wandb-20260912-0159` was exported read-only using the
existing `Down_Identity_Ablations/export_runs.py` exporter with its RUNS mapping
set to L=`olmo3-1b-all-lowrank-linear-fp8-30b`, and Q/K/V/O=
`olmo3-1b-all-linear-{q,k,v,o}-only-silu-fp8-30b`. All five full exports completed
with14307 rows/208 keys each. Credentials were not printed or saved in outputs.
See `CONTROL_RESULTS_20260912.md` for the important distinction between training
improvement and validation performance. Results/raw snapshots remain git-ignored.
