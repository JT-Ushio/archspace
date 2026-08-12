# OLMo3 with MLA, MorphNorm, and Max-Logit FA3

This directory is an external extension tested against:

- OLMo-core commit `064b172e51695259b40684b12ccf10e6e71ad46c`;
- flash-attention-max-logits commit `4aa5fbfb7dcfd59ba3bf9bd81c2f5938b39f7b95`.

The OLMo checkout currently reports version 2.5.0 but contains unreleased APIs, so use the tested
source commit rather than assuming the released PyPI `2.5.0` package is equivalent. The extension
provides:

- native MLA with materialized MHA/GQA training and prefill;
- compressed single-latent-head MQA decoding;
- MorphNorm and its normalization ablations;
- the custom FA3 backend from `flash-attention-max-logits`;
- per-layer, per-query-head max-attention-logit metrics recorded by OLMo and W&B.

The OLMo3 block layout, reordered residual norms, FFN, vocabulary, optimizer, sliding-window
pattern, RoPE theta, and per-layer YaRN scaling are unchanged unless explicitly overridden.

## Model Layout

The default MLA model uses:

| Field | Value |
|---|---:|
| model width | 4096 |
| query heads | 32 |
| training/prefill KV heads | 32 (MHA, same as OLMo3-7B) |
| Q/K head width | 128 |
| NoPE width | 64 |
| RoPE width | 64 |
| value head width | 128 |
| query latent rank | 1024 |
| KV latent rank | 512 |
| decode KV heads | 1 shared latent head |
| resulting model parameters | 6,132,379,648 |

`rope_dim` accepts every even value from `0` through `128`. At `0`, no RoPE module is built and
the compressed cache has a zero-width RoPE-key component. OLMo's RoPE theta and YaRN config remain
authoritative whenever `rope_dim > 0`.

## Attention Paths

- Training and ordinary prefill materialize 32 K/V heads and call the configured FA3 backend.
- Cached prompt attention uses the same materialized path, then stores the raw KV latent and shared
  rotated RoPE key.
- Cached decoding absorbs the K up-projection into Q and applies the V up-projection after latent
  attention. It currently uses PyTorch SDPA with one shared latent K/V head, not FlashMLA.
- The compressed cache stores `kv_lora_rank + rope_dim` elements per token per layer.
- OLMo3's causal SWA/full-attention layer pattern is preserved in both materialized and cached paths.

## Normalization Modes

| Mode | Behavior | MQA decode |
|---|---|---|
| `baseline` | Q/KV latent norms only | supported |
| `materialize_norm` | Q/KV latent norms, per-head Q RMSNorm, shared RoPE-key RMSNorm | supported |
| `standard_qk_norm` | Q/KV latent norms, per-head Q RMSNorm, full materialized K RMSNorm | rejected because the K norm is nonlinear and head-dependent |
| `morphnorm` | Q latent norm, no KV latent norm by default, per-head Q norm, shared RoPE-key norm, MorphNorm on NoPE K | supported |

`norm_type=baseline` means the **MLA baseline**. The true standard-attention OLMo baseline has its
own config below.

### Cross-Layer cKV Residual

Set `use_ckv_layer_residual=true` to pass the unnormalized KV latent between transformer layers:

```text
raw_ckv_i = w_kv_a_i(x_i)[:kv_lora_rank] + raw_ckv_(i-1)
normalized_ckv_i = kv_a_layernorm_i(raw_ckv_i)
```

Layer 0 omits the addition. The default is `false`, and the option adds no parameters or checkpoint
keys. It currently targets OLMo's reordered-norm blocks used by the supplied configs; pipeline
parallelism is rejected because OLMo's pipeline-stage interface does not carry `raw_ckv`.

The reference removal flags map to:

```text
remove_q_a_layernorm=true  -> use_q_a_layernorm=false
remove_kv_a_layernorm=true -> use_kv_a_layernorm=false
```

MorphNorm statistics are accumulated over all gradient-accumulation microbatches and committed at
OLMo's complete-batch boundary. Mandatory dry-runs clear pending statistics without overwriting a
restored checkpoint value.

## Max Attention Logits

The custom FA3 call returns an FP32 tensor with shape `[num_query_heads]`. For layer `l` and query
head `h`, it is:

```text
max over local batch and all valid query/key pairs of
softmax_scale * dot(Q[l, h], K[l, group(h)])
```

The value is computed from FA3's existing softmax row maxima, after causal/SWA/packed-document
masking. It does not materialize `[B,H,T,T]` and is non-differentiable.

This matches Kimi-MuonClip's control granularity: **one value per layer and query head**. The old
MorphNorm Trainer diagnostic collapsed heads into one layer scalar; this implementation preserves
the head dimension and additionally emits summaries.

Across microbatches, each backend accumulates with `maximum` on GPU. After the full batch, OLMo
records every scalar with `ReduceType.max`, so HSDP ranks are reduced in one packed metric
collective rather than one collective per attention forward.

W&B metric names are:

```text
train/block 00/attention max logit/head 00
train/block 00/attention max logit
train/attention max logit
```

The supplied configs enable `MaxLogitsWandBCallback`. Set `WANDB_API_KEY`, and optionally override:

```bash
--trainer.callbacks.wandb.project=my-project \
--trainer.callbacks.wandb.entity=my-entity \
--trainer.callbacks.wandb.group=my-group
```

Max-logit metrics are uploaded every 100 training steps by default while all other W&B metrics
retain their original cadence. Override the max-logit interval independently with:

```bash
--trainer.callbacks.wandb.max_logits_log_interval=500
```

To disable the 1,024 per-head W&B series while retaining layer/model summaries:

```bash
# MLA configs
--model.block.sequence_mixer.log_max_logits_per_head=false

# Standard OLMo baseline config
--model.block.sequence_mixer.log_max_logits_per_head=false
```

`metrics_collect_interval` batches metric communication but does not control W&B cadence. Filtering
the W&B metrics does not skip the max-logit kernel output or distributed metric reduction.

## Server Installation

OLMo-core is assumed to be source-installed. Install this extension without changing that
environment's dependency resolution:

```bash
cd /path/to/OLMo-core
test "$(git rev-parse HEAD)" = "064b172e51695259b40684b12ccf10e6e71ad46c"

python -m pip install -e /path/to/olmo-with-mla-and-morphnorm --no-deps
```

The supplied configs enable W&B and OLMo downstream evaluation. Ensure the server environment has
the corresponding optional dependencies, for example:

```bash
python -m pip install -e '/path/to/OLMo-core[eval,wandb]'
```

For an initial smoke run without those services:

```bash
--trainer.callbacks.wandb.enabled=false \
--trainer.callbacks.lm_evaluator.enabled=false \
--trainer.callbacks.downstream_evaluator.enabled=false
```

Build the local FA3 fork on the H200/CUDA 12.8 server. `FLASH_ATTENTION_FORCE_BUILD=TRUE` is
important because the fork retains upstream's `3.0.0` version and must not download an unpatched
official wheel:

```bash
cd /path/to/flash-attention-max-logits
git checkout 4aa5fbfb7dcfd59ba3bf9bd81c2f5938b39f7b95
git submodule update --init csrc/cutlass
cd /path/to/flash-attention-max-logits/hopper

python -m pip install setuptools wheel packaging ninja einops

FLASH_ATTENTION_FORCE_BUILD=TRUE \
FLASH_ATTENTION_DISABLE_SM80=TRUE \
FLASH_ATTENTION_DISABLE_FP16=TRUE \
FLASH_ATTN_LOCAL_VERSION=maxlogits.4aa5fbf \
MAX_JOBS=16 \
python -m pip install --force-reinstall --no-deps --no-build-isolation .
```

Remove `FLASH_ATTENTION_DISABLE_FP16=TRUE` if FP16 kernels are needed. Verify the installed API:

```bash
python - <<'PY'
import inspect
try:
    from flash_attn_3 import flash_attn_interface as fa3
except ImportError:
    import flash_attn_interface as fa3

assert "return_max_logits" in inspect.signature(fa3.flash_attn_func).parameters
print("custom FA3 max-logit API available")
PY
```

For Mac downloads, use:

```bash
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890
```

The Mac is not expected to compile or run FA3.

Use the source-build command above rather than the fork README's upstream `uv` source example; an
upstream FA3 installation does not contain `return_max_logits` and will be rejected at model build.

## Configs

All executable configs are under `cfgs/` and share the same official OLMo3-7B part-1 recipe:

| Config | Attention variant |
|---|---|
| `OLMo-3-1025-7B-pretrain-1-baseline.py` | true standard OLMo3 attention; 7,298,617,344 params |
| `OLMo-3-1025-7B-pretrain-1-mla-baseline.py` | MLA baseline; 6,132,387,840 params |
| `OLMo-3-1025-7B-pretrain-1-mla-materialize-norm.py` | MLA MaterializeNorm; 6,132,393,984 params |
| `OLMo-3-1025-7B-pretrain-1-mla-standard-qknorm.py` | MLA standard QKNorm; 6,132,396,032 params |
| `OLMo-3-1025-7B-pretrain-1-mla-morphnorm.py` | MLA MorphNorm; 6,132,379,648 params |

The true OLMo baseline is an architecture control, not a parameter/FLOP-matched control. The four
MLA variants are closely size-matched; MorphNorm additionally follows the reference behavior of
omitting the KV latent RMSNorm.

Example:

```bash
torchrun --nproc-per-node=8 \
  /path/to/olmo-with-mla-and-morphnorm/cfgs/OLMo-3-1025-7B-pretrain-1-mla-morphnorm.py \
  --save-folder=/path/to/checkpoints \
  --work-dir=/path/to/work-dir \
  --data-root=/path/to/tokenized-data \
  --name=olmo3-7b-mla-morphnorm \
  --trainer.callbacks.wandb.project=my-project \
  --trainer.callbacks.downstream_evaluator.lazy=true
```

`--data-root` must contain the path layouts required by both `OLMo_mix_0625_official` and
`v3_small_ppl_validation`. Disable the LM/downstream callbacks for a first kernel smoke test if
those validation assets or Hugging Face caches are not available yet.

Additional MLA overrides include:

```bash
--model.block.sequence_mixer.rope_dim=0
--model.block.sequence_mixer.use_q_a_layernorm=false
--model.block.sequence_mixer.use_kv_a_layernorm=false
--model.block.sequence_mixer.use_ckv_layer_residual=true
--model.block.sequence_mixer.q_lora_rank=1536
--model.block.sequence_mixer.kv_lora_rank=512
```

Later stages must load a matching checkpoint from the preceding MLA stage. Official
standard-attention checkpoints are not shape-compatible with MLA projections.

## Source Layout

`attention.py` remains a small compatibility facade for old serialized class paths. The
implementation is split by responsibility:

```text
config.py           MLA config, validation, and parameter counting
module.py           OLMo SequenceMixer module and forward orchestration
attention_paths.py  materialized and absorbed attention paths
morphnorm.py         MorphNorm math and batch statistics
max_logits.py        custom FA3 adapter/backend and standard-attention config
hooks.py             OLMo lifecycle and auxiliary-metric integration
cache.py             compressed MLA cache
patch.py             model-config conversion helpers
```

Projection and normalization attributes remain directly on `MLAAttention`, preserving checkpoint
keys such as `w_q_a`, `w_q_b`, `w_kv_a`, `w_k_b`, `w_v_b`, and `morphnorm_scale`.

## Verification

CPU tests and static checks do not require the custom CUDA extension:

```bash
PYTHONPATH=/path/to/olmo-with-mla-and-morphnorm/src \
pytest -q /path/to/olmo-with-mla-and-morphnorm/tests
```

On H200, also run the fork's max-logit tests:

```bash
cd /path/to/flash-attention-max-logits/hopper
PYTHONPATH="$PWD" pytest -q -s test_flash_attn.py -k max_logits
```

## Current Boundaries

- The supplied dense HSDP topology is supported. Max-logit head naming is not implemented for TP or
  Ulysses head sharding, and custom monitoring rejects context parallelism.
- FA3 max-logit monitoring is active during training. Evaluation uses ordinary FA3 without metric
  overhead; absorbed MLA decoding still uses PyTorch SDPA.
- `standard_qk_norm` is a training/prefill ablation only.
- FP8/quantized transformations of absorbable MLA K/V up-projections are not supported in decoding.
- MorphNorm's distributed ratio currently assumes all ranks execute the same dense layers; pipeline
  parallel training is not supported.
- OLMo's standard Hugging Face exporter does not describe MLA parameters.
