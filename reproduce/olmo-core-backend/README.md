# OLMo-core AHN Architecture Experiments

This directory is a non-invasive OLMo-core extension for OLMo3-1B linear-attention experiments
and a Qwen3.5-style 407M comparison grid using the Dolma2 tokenizer.

The original OLMo3-1B experiments are:

1. replace every sequence-mixing layer with linear attention;
2. preserve OLMo3's `3 × SWA + 1 × Full` layout, replace each SWA layer with linear attention,
   and leave every Full Attention layer unchanged.

GDN is the default. The same recipe can switch to KDA or GDN2 through a config override without
editing OLMo-core or the training script.

The extension is pinned to:

- OLMo-core commit [`b217eb511d5ea191167488a10bb05a38eff53555`](https://github.com/JT-Ushio/OLMo-core-muon-fix/commit/b217eb511d5ea191167488a10bb05a38eff53555)
- flash-linear-attention commit [`35dceaee5408e69a555fec34cb215c93c375dabe`](https://github.com/fla-org/flash-linear-attention/commit/35dceaee5408e69a555fec34cb215c93c375dabe)

## Variants

OLMo3-1B has 16 layers. Its native sliding-window pattern marks layers
`0,1,2,4,5,6,8,9,10,12,13,14` as SWA and layers `3,7,11,15` as Full Attention.

| Recipe | Linear layers | Full Attention layers | Default parameters |
| --- | --- | --- | ---: |
| baseline | none | native 16-layer OLMo3 pattern | 1,484,916,736 |
| all-linear GDN | all 16 | none | 1,754,864,128 |
| hybrid GDN | 12 former SWA layers | 3, 7, 11, 15 | 1,687,377,280 |
| all-linear KDA | all 16 | none | 1,502,613,760 |
| hybrid KDA | 12 former SWA layers | 3, 7, 11, 15 | 1,498,189,504 |
| all-linear GDN2 | all 16 | none | 1,636,307,200 |
| hybrid GDN2 | 12 former SWA layers | 3, 7, 11, 15 | 1,598,459,584 |

The counts use the defaults below. They change when head or value-expansion settings are
overridden.

## Qwen3.5-style 407M grid

The 407M recipes share the following backbone settings:

| Setting | Value |
| --- | ---: |
| layers / model dimension / FFN | 16 / 1,024 / 3,584 |
| sequence length | 8,192 |
| tokenizer / padded vocabulary | Dolma2 / 100,352 |
| Full Attention query / KV heads | 8 / 1 |
| Full Attention head dimension | 256 |
| Full Attention output gate / QK norm | element-wise / per-head Qwen RMSNorm |
| RoPE theta / partial rotary factor | 10,000,000 / 0.25 (64 dimensions) |
| GDN key / value heads | 8 / 16 |
| GDN key / value head dimension | 128 / 128 |
| GDN short convolution | width 4 |
| normalization | Qwen RMSNorm, epsilon 1e-6 |
| embedding / LM-head weights | not tied |

The ten entry points are the Cartesian product of five layer layouts and two training profiles:

| Layout | Layer pattern | Total params | Non-embedding params |
| --- | --- | ---: | ---: |
| `all-gdn` | 16 GDN | 516,722,176 | 413,961,728 |
| `gdn3-full1` | repeat 3 GDN + 1 Full | 510,235,520 | 407,475,072 |
| `all-full` | 16 Full | 490,775,552 | 388,015,104 |
| `swa3-full1` | repeat 3 4K-SWA + 1 Full | 490,775,552 | 388,015,104 |
| `all-swa` | 16 4K-SWA | 490,775,552 | 388,015,104 |

The exact `510,235,520 / 407,475,072` target is obtained by the `gdn3-full1` main backbone with
untied input embeddings and LM head. The supplied target does not include an MTP auxiliary
module. The pinned OLMo-core has no MTP prediction head or shifted-token auxiliary-loss path, so
these recipes intentionally do **not** claim to train MTP. Adding two inert config fields would be
misleading; MTP needs a separate model and train-module implementation before it can be enabled.

| Profile | Global token batch | Maximum steps | Maximum training tokens |
| --- | ---: | ---: | ---: |
| `256K` | 262,144 | 3,915 | 1,026,293,760 |
| `512K` | 524,288 | 6,526 | 3,421,503,488 |

Both profiles use Muon with `lr_max=2e-3`, `weight_decay=0.1`, `mu=0.95`,
`adjust_lr=rms_norm`, and `use_polar_express=false`. Every model uses
`InitMethod.scaled_truncated_normal`. The schedule is token-based WSD: linear warmup for
33,554,432 tokens, a stable peak, then a 1-sqrt decay over the final 10% of training to
`2e-4 = 0.1 * lr_max`. For decay progress `r` in `[0, 1]`, the tail is
`lr = lr_max * (1 - sqrt(r)) + lr_final * sqrt(r)`.

## Implementation

```text
src/olmo_ahn/linear_attention.py  serializable GDN/KDA/GDN2 config and FLA adapter
src/olmo_ahn/patch.py             non-mutating OLMo3 TransformerConfig patches
src/olmo_ahn/optim.py             serializable Muon config with native parameter grouping
cfgs/_models.py                   baseline, all-linear, and hybrid model builders
cfgs/_pretrain_common.py          shared 150B-sample stage-1 recipe
cfgs/OLMo3-1B-stage1-*.py         three direct training entry points
cfgs/_qwen35_407m_models.py       five Qwen3.5-style 407M backbone builders
cfgs/_qwen35_407m_pretrain.py     shared Muon/data/scheduler recipe
cfgs/Qwen3.5-407M-*.py            ten explicit comparison entry points
tests/                             config, serialization, pattern, and GPU checks
```

`patch_all_linear_attention()` replaces the base sequence mixer. `patch_swa_with_linear_attention()`
copies the configuration, installs a linear base mixer, and creates overrides only for the four
original Full Attention layers. The input config is never mutated.

GDN uses the native `GatedDeltaNet` implementation in the pinned OLMo-core. KDA and GDN2 use the
pinned FLA layers behind a small adapter that implements OLMo's `SequenceMixer` build,
initialization, FLOP-reporting, and packed-document interfaces. Packed-document cumulative lengths
are forwarded to FLA as `cu_seqlens`, so recurrent and convolution states reset at document
boundaries.

Defaults:

| Setting | GDN | KDA | GDN2 |
| --- | ---: | ---: | ---: |
| query/key heads | 16 | 16 | 16 |
| value heads | 16 | 16 | 16 |
| head dimension | 128 | 128 | 128 |
| value expansion | 2.0 | 1.0 | 1.0 |
| short convolution | enabled, width 4 | enabled, width 4 | enabled, width 4 |
| negative eigenvalues | disabled | disabled | disabled |

KDA/GDN2 support OLMo's `InitMethod.normal` and `InitMethod.scaled_truncated_normal`. The recipe
uses HSDP, not tensor or context parallelism. Float8 is disabled. With the pinned OLMo-core,
three-dimensional convolution weights are routed to AdamW; vector parameters such as `A_log` and
`dt_bias` use AdamW without weight decay.

`SerializableMuonConfig` inherits the backend's `flatten=False` default, preserving per-head
Muon for Full/SWA attention projections. Do not set `flatten=True`: Dion rejects it for
`num_heads > 1`, and convolution weights already use AdamW.

## Local Mac Verification

No training, CUDA compilation, or FLA kernel execution is required on the Mac. Point the tests to
the pinned framework checkout:

```bash
export OLMO_CORE=/path/to/OLMo-core-muon-fix
test "$(git -C "$OLMO_CORE" rev-parse HEAD)" = \
  "b217eb511d5ea191167488a10bb05a38eff53555"

PYTHONPATH="$PWD/src:$OLMO_CORE/src" python3 -m pytest -q
PYTHONPATH="$PWD/src:$OLMO_CORE/src" python3 -m ruff check .
git diff --check
```

The CUDA tests are skipped locally. They build the three mixers, compare analytic and actual
parameter counts, exercise packed-document state resets, and check finite forward, backward, and
input/parameter gradients.
The optimizer test also checks per-head grouping and convolution routing locally. On a CUDA
server with Dion installed, it executes a first optimizer step with both parameter types:

```bash
python3 -m pytest -q tests/test_optim.py
```

## Server Installation

Install the pinned OLMo-core first, then FLA with its CUDA dependencies. Install this extension
last with `--no-deps` so pip cannot replace either pinned checkout.

```bash
git clone --branch ready_for_archspace_base \
  https://github.com/JT-Ushio/OLMo-core-muon-fix.git /path/to/OLMo-core
cd /path/to/OLMo-core
git checkout b217eb511d5ea191167488a10bb05a38eff53555
python3 -m pip install -e '.[eval,wandb,dion]'
python3 -m pip install flash-attn --no-build-isolation

python3 -m pip uninstall -y fla-core flash-linear-attention
python3 -m pip install --no-build-isolation \
  'flash-linear-attention[cuda] @ git+https://github.com/fla-org/flash-linear-attention.git@35dceaee5408e69a555fec34cb215c93c375dabe'

python3 -m pip install -e /path/to/AHN/reproduce/olmo-core-backend --no-deps
cd /path/to/AHN/reproduce/olmo-core-backend
```

Before a training run, validate all three kernels on the target GPU:

```bash
PYTHONPATH="$PWD/src:/path/to/OLMo-core/src" \
  python3 -m pytest -q tests/test_gpu_linear_attention.py
```

This is the minimum CUDA correctness gate. It does not establish throughput or long-run training
stability; record those separately on the actual server configuration.

## Configuration Dry Runs

The model scripts use OLMo's native `main(build_config)` CLI. A dry run builds and serializes the
complete recipe without loading data or starting training:

```bash
python3 cfgs/OLMo3-1B-stage1-all-linear.py \
  --dry-run \
  --save-folder=/tmp/ahn-config \
  --name=olmo3-1b-all-linear-gdn

python3 cfgs/OLMo3-1B-stage1-hybrid-linear.py \
  --dry-run \
  --save-folder=/tmp/ahn-config \
  --name=olmo3-1b-hybrid-gdn \
  --model.block.sequence_mixer.attention_type=gdn2

python3 cfgs/Qwen3.5-407M-gdn3-full1-256K.py \
  --dry-run \
  --save-folder=/tmp/ahn-qwen35-config \
  --name=qwen35-407m-gdn3-full1-256k
```

All ten `cfgs/Qwen3.5-407M-{layout}-{profile}.py` files have the same CLI. Their architecture and
training-profile choices are fixed in the filename so comparison runs cannot silently drift.

## Training

Default GDN, all layers:

```bash
torchrun --nproc-per-node=8 \
  cfgs/OLMo3-1B-stage1-all-linear.py \
  --save-folder=/path/to/checkpoints/all-gdn \
  --work-dir=/path/to/work/all-gdn \
  --data-root=/path/to/tokenized-data \
  --name=olmo3-1b-all-gdn
```

Default GDN, 12 linear layers plus four unchanged Full Attention layers:

```bash
torchrun --nproc-per-node=8 \
  cfgs/OLMo3-1B-stage1-hybrid-linear.py \
  --save-folder=/path/to/checkpoints/hybrid-gdn \
  --work-dir=/path/to/work/hybrid-gdn \
  --data-root=/path/to/tokenized-data \
  --name=olmo3-1b-hybrid-gdn
```

Switch either entry point to KDA:

```bash
--model.block.sequence_mixer.attention_type=kda
```

Switch either entry point to GDN2:

```bash
--model.block.sequence_mixer.attention_type=gdn2
```

The linear mixer can also be tuned with native config overrides:

```bash
--model.block.sequence_mixer.n_heads=16 \
--model.block.sequence_mixer.n_v_heads=32 \
--model.block.sequence_mixer.head_dim=128 \
--model.block.sequence_mixer.expand_v=1.0 \
--model.block.sequence_mixer.conv_size=4 \
--model.block.sequence_mixer.allow_neg_eigval=false
```

These settings express the Qwen3.5-style linear-attention dimensions exactly. The names differ
because AHN uses one algorithm-neutral config for GDN, KDA, and GDN2:

| Qwen3.5 field | AHN field or expression |
| --- | --- |
| `linear_conv_kernel_dim=4` | `conv_size=4` |
| `linear_key_head_dim=128` | `head_dim=128` |
| `linear_num_key_heads=16` | `n_heads=16` |
| `linear_num_value_heads=32` | `n_v_heads=32` |
| `linear_value_head_dim=128` | `head_dim * expand_v = 128`, so `expand_v=1.0` |

The Qwen3.5 JSON field names are not accepted directly by the OLMo-core CLI. Use the AHN names
above. `n_v_heads` must be greater than or equal to `n_heads` and divisible by it, and
`head_dim * expand_v` must be an integer.

KDA additionally exposes its bounded safe-gate path:

```bash
--model.block.sequence_mixer.attention_type=kda \
--model.block.sequence_mixer.safe_gate=true \
--model.block.sequence_mixer.lower_bound=-5.0
```

The baseline uses the identical tokenizer, data mix, batch settings, HSDP/Muon recipe, evaluators,
and logging callbacks:

```bash
torchrun --nproc-per-node=8 \
  cfgs/OLMo3-1B-stage1-baseline.py \
  --save-folder=/path/to/checkpoints/baseline \
  --work-dir=/path/to/work/baseline \
  --data-root=/path/to/tokenized-data \
  --name=olmo3-1b-baseline
```

For example, launch the 407M Qwen3.5-style hybrid with the 256K-token profile using:

```bash
torchrun --nproc-per-node=8 \
  cfgs/Qwen3.5-407M-gdn3-full1-256K.py \
  --save-folder=/path/to/checkpoints/qwen35-407m-gdn3-full1-256k \
  --work-dir=/path/to/work/qwen35-407m-gdn3-full1-256k \
  --data-root=/path/to/tokenized-data \
  --name=qwen35-407m-gdn3-full1-256k
```

## Compatibility Boundaries

- The three linear algorithms have different parameters and checkpoint keys; their checkpoints
  are not interchangeable.
- The all-linear and hybrid variants are not checkpoint-compatible with each other.
- Full Attention blocks in the hybrid model retain the baseline parameter names and topology, but
  the complete model checkpoint still differs because the other 12 blocks changed.
- Tensor parallelism is not enabled. KDA/GDN2 context parallelism is not implemented in this
  extension. The supplied HSDP recipe is the supported distributed path.
- OLMo's built-in Hugging Face converter does not know these custom sequence-mixer configs. A
  dedicated HF architecture and checkpoint converter are required before publication.
- MTP is not implemented by the pinned OLMo-core backend and is not active in the 407M recipes.
