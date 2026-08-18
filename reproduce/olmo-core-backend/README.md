# OLMo3-1B Cubit

This directory is an external OLMo-core extension for the Cubit architecture proposed in
[InternLM/archspace#20](https://github.com/InternLM/archspace/issues/20) and
[arXiv:2605.06501](https://arxiv.org/html/2605.06501). Its layout follows the MoSE reproduction
backend: a pinned framework dependency, an importable extension package, non-mutating model
patches, matched recipes, and CPU integration tests.

It is tested against commit `f38e580063e48aa4b212837210df2e2038e80148` from
[`JT-Ushio/OLMo-core-muon-fix`](https://github.com/JT-Ushio/OLMo-core-muon-fix), branch
`ready_for_archspace_base`.

## Implementation

For each head, Cubit computes the paper's reference kernel and LRR correction:

```python
normalized_r = normalize(r) * reference_scale
inverse_sigma = masked_softmax(r @ normalized_r.T) + exp(log_regularization) * I
scale = lrr_lower + lrr_range * sigmoid(w_lrr(x))
solution = solve_triangular(inverse_sigma, scale * v)
output = attention(q, k, solution)
```

`inverse_sigma` denotes the matrix that is inverted by the solve. Thus `solution` is
`Sigma @ S @ V`, and the last line implements `A @ Sigma @ S @ V` from equation 13. The causal
softmax makes `inverse_sigma` lower triangular. The streaming implementation solves it one
diagonal block at a time rather than explicitly forming a matrix inverse or a complete kernel
matrix.

The paper's Appendix H contains several typographical inconsistencies (`LRR`/`lrr` and
`casual_mask`/`causal_mask`). This implementation uses consistent names and tests the complete
formula against an independent reference computation.

### Numerical behavior

- Kernel scores, masked softmax, LRR, ridge regularization, and the triangular solve use FP32
  tensors. The training recipe enables TF32 tensor cores for the batched kernel GEMMs while
  retaining FP32 accumulation.
- `regularization` is stored in log space and exponentiated in the forward pass.
- Reference normalization uses a configurable epsilon (`1e-6` by default) to avoid division by
  zero.
- Causal and sliding-window masks are applied to both the KRR kernel and final aggregation.
- Packed documents are separated in the KRR matrix; FlashAttention receives the same cumulative
  document lengths for the final aggregation.
- Q, K, and the KRR solution are made last-dimension-contiguous at the FlashAttention boundary;
  `torch.linalg.solve_triangular` does not guarantee the layout required by FA3.

### Reference projection

`share_reference=false` is the default, matching Appendix H and the paper's stronger independent
reference-projection result. It adds one `d_model -> n_heads * head_dim` projection per Cubit layer.
Set `share_reference=true` for the shared-key ablation; the reference then uses OLMo's actual
post-QK-norm key representation.

| Config | Reference | Parameters |
|---|---|---:|
| `OLMo3-1B-stage1-baseline.py` | n/a | 1,484,916,736 |
| `OLMo3-1B-stage1-cubit.py` | independent | 1,552,550,912 |
| Cubit with `share_reference=true` | shared key | 1,485,442,048 |

## Streaming KRR kernel

The default `krr_implementation=streaming` path applies the FlashAttention idea to Cubit's causal
linear system. For each query block it computes only a `block_size x prefix_length` kernel tile,
subtracts the contribution of already solved tokens, and solves the small diagonal triangular
block. Its custom backward first performs the corresponding transposed block solve, then
recomputes kernel tiles to obtain exact softmax and reference gradients.

Consequently, Cubit no longer saves or materializes a complete
`(batch, heads, sequence, sequence)` tensor. Per head, the KRR solve uses
`O(T * D + T * block_size)` activation memory instead of `O(T^2)`. The exact exponential kernel
still requires `O(T^2 * D)` arithmetic; unlike linear-attention approximations, this optimization
does not change the paper's formula.

The final `attention(q, k, solution)` remains dispatched through OLMo-core's
`AttentionBackendName.flash_3` backend. A dense KRR implementation is retained only as a numerical
reference and can be selected with `krr_implementation=dense`.

Measure the isolated KRR forward and backward on the target GPU with:

```bash
PYTHONPATH=src python3 benchmarks/benchmark_krr.py \
  --implementation=streaming --seq-len=4096 --block-size=64
```

Use a shorter sequence such as 512 or 1024 when benchmarking `--implementation=dense`; its full
FP32 kernel matrix can exhaust device memory at the training sequence length.

## Layout

```text
src/olmo_cubit/attention.py   Cubit module and serializable OLMo config
src/olmo_cubit/krr.py         streaming causal KRR solve and custom backward
src/olmo_cubit/patch.py       non-mutating TransformerConfig patch
src/olmo_cubit/optim.py       serializable Muon parameter-group integration
cfgs/_models.py               OLMo3-1B baseline and Cubit builders
cfgs/_pretrain_common.py      shared stage-1 Muon/HSDP recipe
cfgs/OLMo3-1B-stage1-*.py     baseline and Cubit entry points
tests/                        CPU formula, masking, model, and recipe tests
```

## Local verification on macOS

No CUDA build or training is required. Use the pinned OLMo-core checkout as a source dependency:

```bash
export OLMO_CORE=/path/to/OLMo-core-muon-fix
test "$(git -C "$OLMO_CORE" rev-parse HEAD)" = \
  "f38e580063e48aa4b212837210df2e2038e80148"

cd /path/to/archspace/reproduce/olmo-core-backend
PYTHONPATH="$PWD/src:$OLMO_CORE/src" python3 -m pytest -q
PYTHONPATH="$PWD/src:$OLMO_CORE/src" python3 -m ruff check .
```

CPU model construction automatically falls back from the configured FlashAttention 3 backend to
OLMo's torch path. The tests do not train a model or require tokenized data.

## Server installation

Install the exact OLMo-core source first, followed by this extension without dependency
replacement:

```bash
git clone --branch ready_for_archspace_base \
  https://github.com/JT-Ushio/OLMo-core-muon-fix.git /path/to/OLMo-core
git -C /path/to/OLMo-core checkout f38e580063e48aa4b212837210df2e2038e80148
python3 -m pip install -e '/path/to/OLMo-core[eval,wandb,dion]'

python3 -m pip install -e \
  /path/to/archspace/reproduce/olmo-core-backend --no-deps
```

For H100/H800, install the same FlashAttention 3 revision pinned by the OLMo-core Dockerfile. CUDA
12.3 or newer is required; CUDA 12.8 is recommended by FlashAttention:

```bash
git clone --recursive https://github.com/Dao-AILab/flash-attention.git \
  /path/to/flash-attention
git -C /path/to/flash-attention checkout 92ca9da8d66f7b34ff50dc080ec0fef9661260d6
git -C /path/to/flash-attention submodule update --init --recursive
cd /path/to/flash-attention/hopper
FLASH_ATTENTION_DISABLE_FP16=TRUE \
FLASH_ATTENTION_DISABLE_SM80=TRUE \
MAX_JOBS=16 \
python3 setup.py install
```

Verify that OLMo sees the backend before launching a run:

```bash
python3 -c 'from olmo_core.nn.attention import AttentionBackendName; AttentionBackendName.flash_3.assert_supported()'
```

On hardware where this pinned FA3 backend is unavailable, install FlashAttention 2 and override
both recipes with `--model.block.sequence_mixer.backend=flash_2` for a matched comparison.

## Training

Run from `/path/to/archspace/reproduce/olmo-core-backend`. Both entry points share the Dolma2
tokenizer, `OLMo_mix_0625_150Bsample`, Muon settings, global token batch, evaluators, checkpoints,
and W&B project.

```bash
torchrun --nproc-per-node=8 \
  cfgs/OLMo3-1B-stage1-baseline.py \
  --save-folder=/path/to/checkpoints/baseline-fa3 \
  --work-dir=/path/to/work/baseline-fa3 \
  --data-root=/path/to/tokenized-data \
  --name=olmo3-1b-baseline-fa3

torchrun --nproc-per-node=8 \
  cfgs/OLMo3-1B-stage1-cubit.py \
  --save-folder=/path/to/checkpoints/cubit-fa3 \
  --work-dir=/path/to/work/cubit-fa3 \
  --data-root=/path/to/tokenized-data \
  --name=olmo3-1b-cubit-fa3
```

`--data-root` must provide the layouts required by `OLMo_mix_0625_150Bsample` and
`v3_small_ppl_validation`.

The correctness-first recipes leave `torch.compile` disabled while continuing to use the
FlashAttention 3 backend. Once eager-mode training is validated, opt into compilation with
`--train-module.compile-model=true`. Keep `TORCHINDUCTOR_CACHE_DIR` on node-local storage and
use a fresh directory after an interrupted run; a persistent shared directory can retain
partially written FX-graph cache entries.

For an initial smoke test, reduce sequence length and disable external callbacks:

```bash
torchrun --nproc-per-node=8 \
  cfgs/OLMo3-1B-stage1-cubit.py \
  --sequence-length=128 \
  --save-folder=/path/to/checkpoints/cubit-smoke \
  --work-dir=/path/to/work/cubit-smoke \
  --data-root=/path/to/tokenized-data \
  --name=olmo3-1b-cubit-smoke \
  --trainer.callbacks.wandb.enabled=false \
  --trainer.callbacks.lm_evaluator.enabled=false \
  --trainer.callbacks.downstream_evaluator.enabled=false
```

### Architecture overrides

Use Cubit only in layers 2 through 14 of the 16-layer model:

```bash
--cubit-start-layer=2 --cubit-end-layer=15
```

Run the shared-key ablation:

```bash
--model.block.sequence_mixer.share_reference=true
```

Override KRR/LRR settings:

```bash
--model.block.sequence_mixer.regularization=1e-6 \
--model.block.sequence_mixer.lrr_lower=0.5 \
--model.block.sequence_mixer.lrr_upper=2.0 \
--model.block.sequence_mixer.krr_block_size=64
```

Use `krr_block_size=32` to reduce peak temporary memory or `128` to trade more memory for fewer
kernel launches. For correctness comparisons only, set
`--model.block.sequence_mixer.krr_implementation=dense`.

The layer range is zero-based and half-open. Dotlist overrides on
`model.block.sequence_mixer.*` affect Cubit layers; dense layers created by a partial range retain
the original OLMo attention config.

## Current limitations

- Tensor parallelism and context parallelism raise explicit `NotImplementedError`; the supplied
  recipe uses HSDP only.
- KV-cached decoding is not implemented because KRR coefficients depend on the complete prefix.
- Float8 is disabled. KRR correction state and triangular solves remain FP32.
- The native OLMo Hugging Face converter does not encode Cubit parameters or computation. A
  dedicated architecture and checkpoint converter are required before publishing weights.
