# Paired initialization ablation

Architecture: rank512 for Q/K/V/O/Gate/Up/Down, no sharing, no bypass;
SiLU in Q/K/V/O/Gate/Up and nonlearnable RMSNorm only in Down. The primary
SwiGLU gate and outer Transformer norms are unchanged.

Write each projection as `D phi(E x)` (historical U/V names differ between
attention and FFN). For each projection independently, sample a dense FP32
W0 with the existing normal-init std and ±3std truncation, then exact reduced
SVD W0=PΣQᵀ. Keep512 singular components; E=√ΣQᵀ, D=P√Σ. Fix each singular
vector sign by requiring its largest-absolute encoder entry to be positive.

| Condition | Factors | Role |
|---|---|---|
| independent | existing initialization, unchanged | paired full-training A |
| svd_raw | balanced truncated SVD | step0 diagnostic B |
| svd_scale_matched | separately match E/D RMS to original initialized factors | paired full-training C |

With SiLU, `D phi(E x)` is NOT the rank512 dense approximation applied to x.
Matched DE is a scaled version of the truncated W0. Matching weight RMS does
not equalize activation distributions. Random W0 contains no pretrained
knowledge; this tests initialization geometry, not pretrained compression.

Preparation (`run_step0_probe.sh`, arch1×8) partitions all112 projection SVDs
and32 fixed held-out8K sequences across8 GPUs. Compare BF16/FP8 initial CE,
factor/product spectra and per-layer a/z/y statistics. No optimizer updates.
Output lives in `scripts/analysis/Initialization_Analysis/results/` and refuses
to overwrite an existing directory. Training requires the verified manifest
and a successful probe marker. Each training process first executes normal
initialization and checks baseline factor hashes, preserving other parameters
and generator consumption, then loads the immutable factors.

Full training: `run_independent_fp8.sh` and `run_svd_scale_matched_fp8.sh`, each
arch2×8 by default. GlobalBS256, seq8192, microbatch4/GPU (accumulate4),30B
tokens, MuonLR.005, FP8, WSD2000-step warmup/final20%decay; checkpoint every1788
steps (~12.5%), no periodic retention deletion, final save additional as usual.
Model init seed remains the baseline default; separate dense-SVD/data seed34521
is recorded in the manifest. Compare matched-token validation CE first, plus
train CE, gradient norms, divergence incidents and wall-clock throughput.

Do not launch raw-SVD full training solely because its step0 CE is lower.
Step0 CE is diagnostic, not evidence of convergence. Additional paired seeds
are needed before attributing small endpoint differences to initialization.

Checkpoint readers should construct the baseline TransformerConfig after
removing `factor_init` and `factor_init_artifacts`; loaded checkpoint weights
must not be reinitialized. Runtime trainer resume keeps the normal checkpoint
restore order (artifact initialization first, checkpoint restore afterward).

## Completed-training comparison

The current authorized A/C submissions use4×8 each, accumulation2; all training
hyperparameters above remain unchanged. See RUN_STATUS.md for actual job IDs.

After both jobs are terminal, synchronize their exact offline W&B segments
through `wandb_sjy` on the gateway and verify upload completion. Then locally:

```bash
python scripts/analysis/Initialization_Analysis/export_training.py --output-dir scripts/analysis/Initialization_Analysis/results/training-final-SNAPSHOT
python scripts/analysis/Initialization_Analysis/compare_training.py scripts/analysis/Initialization_Analysis/results/training-final-SNAPSHOT
```

Use an already authenticated environment or the existing exporter's authorized
`--credential-file` argument; never print or paste keys. These are CPU-only
read-only W&B export/local-analysis commands, not gateway computation.
The comparator rejects unfinished/short runs, unexpected config differences,
duplicate or mismatched steps/tokens, nonfinite CE, and missing validation
sources/points. Outputs include raw every-step train CE, all source validation
CEs, endpoint tables, last100/500/1000-step means, and three SVG curves. The
macro is equal-source CE, not token-weighted CE/BPB; single-seed differences
are descriptive. Preserve snapshots and the separate checkpoint/probe ledger.
