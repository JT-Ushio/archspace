# OLMo3-1B GDN/KDA/GDN2 Attention

This directory is a non-invasive OLMo-core extension for two OLMo3-1B linear-attention
experiments:

1. replace every sequence-mixing layer with linear attention;
2. preserve OLMo3's `3 × SWA + 1 × Full` layout, replace each SWA layer with linear attention,
   and leave every Full Attention layer unchanged.

GDN is the default. The same recipe can switch to KDA or GDN2 through a config override without
editing OLMo-core or the training script.

The extension is pinned to:

- OLMo-core commit [`f38e580063e48aa4b212837210df2e2038e80148`](https://github.com/JT-Ushio/OLMo-core-muon-fix/commit/f38e580063e48aa4b212837210df2e2038e80148)
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

## Implementation

```text
src/olmo_ahn/linear_attention.py  serializable GDN/KDA/GDN2 config and FLA adapter
src/olmo_ahn/patch.py             non-mutating OLMo3 TransformerConfig patches
src/olmo_ahn/optim.py             serializable Muon groups with conv flattening
cfgs/_models.py                   baseline, all-linear, and hybrid model builders
cfgs/_pretrain_common.py          shared 150B-sample stage-1 recipe
cfgs/OLMo3-1B-stage1-*.py         three direct training entry points
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

KDA/GDN2 support OLMo's `InitMethod.normal`, which is the native OLMo3-1B setting. The recipe uses
HSDP, not tensor or context parallelism. Float8 is disabled. Muon flattens depthwise-convolution
weights, while `A_log` and `dt_bias` use AdamW without weight decay.

## Local Mac Verification

No training, CUDA compilation, or FLA kernel execution is required on the Mac. Point the tests to
the pinned framework checkout:

```bash
export OLMO_CORE=/path/to/OLMo-core-muon-fix
test "$(git -C "$OLMO_CORE" rev-parse HEAD)" = \
  "f38e580063e48aa4b212837210df2e2038e80148"

PYTHONPATH="$PWD/src:$OLMO_CORE/src" python3 -m pytest -q
PYTHONPATH="$PWD/src:$OLMO_CORE/src" python3 -m ruff check .
git diff --check
```

The CUDA tests are skipped locally. They build the three mixers, compare analytic and actual
parameter counts, exercise packed-document state resets, and check finite forward, backward, and
input/parameter gradients.

## Server Installation

Install the pinned OLMo-core first, then FLA with its CUDA dependencies. Install this extension
last with `--no-deps` so pip cannot replace either pinned checkout.

```bash
git clone --branch ready_for_archspace_base \
  https://github.com/JT-Ushio/OLMo-core-muon-fix.git /path/to/OLMo-core
cd /path/to/OLMo-core
git checkout f38e580063e48aa4b212837210df2e2038e80148
python3 -m pip install -e '.[eval,wandb,dion]'
python3 -m pip install flash-attn --no-build-isolation

python3 -m pip uninstall -y fla-core flash-linear-attention
python3 -m pip install \
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
```

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
