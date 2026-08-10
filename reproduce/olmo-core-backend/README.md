# OLMo3-1B MoSE-SwiGLU

This directory is an external OLMo-core extension for the MoSE architecture experiments. The
implementation combines fixed-scalar SwiGLU channel control with Mixture of Subspace Experts
(MoSE) gate, up, and optional down projections.

It is tested against commit `f38e580063e48aa4b212837210df2e2038e80148` from
[`JT-Ushio/OLMo-core-muon-fix`](https://github.com/JT-Ushio/OLMo-core-muon-fix), branch
`ready_for_archspace_base`.

## Implemented Variants

For the three native-topology controls, OLMo names the gate, down, and up projections `w1`, `w2`,
and `w3`, respectively. Those modules and checkpoint keys remain unchanged.

### Standard SwiGLU

```python
gate = self.w1(x)
up = self.w3(x)
hidden = F.silu(gate) * up
return self.w2(hidden)
```

The baseline config uses OLMo's native `FeedForwardConfig` and `FeedForward` without patching it.

### SiTU

```python
gate = self.w1(x)
up = self.w3(x)

gate_linear = self.beta_gate * torch.tanh(gate / self.beta_gate)
up = self.beta_up * torch.tanh(up / self.beta_up)
hidden = gate_linear * torch.sigmoid(gate) * up

return self.w2(hidden)
```

### Asymmetric RationalClip

```python
gate = self.w1(x)
up = self.w3(x)

up = up * torch.rsqrt(1.0 + (up / self.beta_up).square())
gate_linear = gate * torch.rsqrt(
    1.0 + (F.relu(gate) / self.beta_gate).square()
)
hidden = gate_linear * torch.sigmoid(gate) * up

return self.w2(hidden)
```

Both controlled variants default to:

```text
beta_gate = 4.0
beta_up = 25.0
```

These values are fixed, non-learned Python scalars and are not configuration options. They add no
parameters or checkpoint entries.
Asymmetric RationalClip caps the up channel symmetrically and caps only the positive part of the
gate's linear factor; the sigmoid still suppresses the uncapped negative tail.

Both controlled nonlinearities and their product are evaluated in FP32 when the projections are
FP16 or BF16, then cast back before `w2`. RationalClip uses an algebraically equivalent reciprocal
form above each cap to avoid overflow while retaining the stated function and positive derivative.

### MoSE-SwiGLU

Gate and up share each low-dimensional U projection, then use independent V projections:

```python
linear_latent = linear_u(x)
nonlinear_latent = nonlinear_u(x)

gate = gate_linear_v(linear_latent) + gate_nonlinear_v(
    apply_nonlinearity(nonlinear_latent, gate_nonlinearity)
)
up = up_linear_v(linear_latent) + up_nonlinear_v(
    apply_nonlinearity(nonlinear_latent, up_nonlinearity)
)
```

`gate_nonlinearity`, `up_nonlinearity`, and `down_nonlinearity` are independently configurable as
`silu` or parameter-free `rms_norm`. All three default to `silu`, preserving the
original behavior and checkpoint topology. RMSNorm is evaluated over the final latent dimension
with `eps=1e-5`, using FP32 math for FP16/BF16 inputs.

The selected SiTU or Asymmetric RationalClip function is applied after the two experts have been
summed into the final gate and up channels. The fixed `4.0/25.0` constants and FP32 control math are
identical to the native-topology experiments.

The down projection uses the corresponding two-expert sum when either down rank is positive:

```python
out = down_linear_v(down_linear_u(hidden))
out += down_nonlinear_v(
    apply_nonlinearity(down_nonlinear_u(hidden), down_nonlinearity)
)
```

If both down ranks are zero, it instead uses one dense `w_down` projection. All four ranks are
configuration fields with these defaults:

```text
r1 = 880
r2 = 880
down_r1 = 880
down_r2 = 880
```

At the defaults, OLMo3-1B has `1,487,013,888` parameters, compared with `1,484,916,736` for the
native model. Each low-rank approximation uses `R = r1 + r2` (and
`R = down_r1 + down_r2` for down). OLMo's deterministic `InitMethod.normal` initializes both the
linear and nonlinear U factors with `TruncNormal(0, 0.02)`, and all corresponding V factors with
`TruncNormal(0, 1 / sqrt(R))`, so the corresponding summed U/V matrix has target standard
deviation `0.02`. A dense `w_down` fallback continues to use `TruncNormal(0, 0.02)`.

## Ablations

| Config | Feed-forward topology | Channel control | Parameters |
|---|---|---|---:|
| `OLMo3-1B-stage1-baseline.py` | native | standard SwiGLU | 1,484,916,736 |
| `OLMo3-1B-stage1-situ.py` | native | SiTU | 1,484,916,736 |
| `OLMo3-1B-stage1-asymmetric-rational-clip.py` | native | Asymmetric RationalClip | 1,484,916,736 |
| `OLMo3-1B-stage1-mose-rmsnorm-silu.py` | MoSE, RMSNorm gate/up + SiLU down | standard SwiGLU | 1,487,013,888 |
| `OLMo3-1B-stage1-mose-situ.py` | MoSE | SiTU | 1,487,013,888 |
| `OLMo3-1B-stage1-mose-asymmetric-rational-clip.py` | MoSE | Asymmetric RationalClip | 1,487,013,888 |

## Layout

```text
src/olmo_mose/feed_forward.py  native control and MoSE modules/configs
src/olmo_mose/hooks.py         deterministic OLMo initialization integration
src/olmo_mose/optim.py         serializable Muon parameter-group integration
src/olmo_mose/patch.py         non-mutating TransformerConfig patches
cfgs/_models.py                official OLMo3-1B model builders
cfgs/_pretrain_common.py       shared 150B-sample Muon recipe
cfgs/OLMo3-1B-stage1-*.py      six ablation entry points
tests/                         CPU formula, config, and integration tests
```

Native SiTU and RationalClip checkpoints are strictly compatible with the native baseline. MoSE
checkpoints use semantic U/V parameter names and are not compatible with native `w1/w2/w3`
checkpoints. MoSE checkpoints are strictly compatible across controls only when all four ranks and
the down topology match.

OLMo's built-in Hugging Face converter does not encode either custom channel-control function and
does not map MoSE U/V weights. Do not use it for these checkpoints: native controlled checkpoints
would be exported as ordinary SwiGLU, while MoSE conversion fails on unmapped keys. A dedicated HF
architecture and converter are required later for publication.

The supplied recipe uses HSDP, and every two-dimensional U/V weight is assigned to Muon. MoSE
tensor parallelism is intentionally rejected because its two-stage shared projections require a
separate distributed plan; the pinned Muon implementation also rejects TP. Float8 is disabled in
the supplied recipe. `SerializableMuonConfig` discards generated optimizer group overrides after
optimizer construction so runtime-saved experiment configs remain loadable.

## Local Verification

No training or CUDA build is required on the Mac. With the framework checkout available locally:

```bash
export OLMO_CORE=/path/to/OLMo-core-muon-fix
test "$(git -C "$OLMO_CORE" rev-parse HEAD)" = \
  "f38e580063e48aa4b212837210df2e2038e80148"

PYTHONPATH="$PWD/src:$OLMO_CORE/src" python3 -m pytest -q
PYTHONPATH="$PWD/src:$OLMO_CORE/src" python3 -m ruff check .
```

## Server Installation

Install the selected OLMo-core source checkout first, then install this extension without allowing
pip to replace it:

```bash
git clone --branch ready_for_archspace_base \
  https://github.com/JT-Ushio/OLMo-core-muon-fix.git /path/to/OLMo-core
cd /path/to/OLMo-core
git checkout f38e580063e48aa4b212837210df2e2038e80148
python3 -m pip install -e '.[eval,wandb,dion]'
python3 -m pip install flash-attn --no-build-isolation

python3 -m pip install -e /path/to/archspace/reproduce/olmo-core-backend --no-deps
```

FlashAttention is intentionally installed as a separate command because its source package needs
PyTorch to be present and must be built with `--no-build-isolation`.

## Training

Run these commands from `/path/to/archspace/reproduce/olmo-core-backend`. The six entry points
share the same tokenizer, data recipe, Muon settings, batch size, evaluator callbacks, and W&B
project.

```bash
torchrun --nproc-per-node=8 \
  cfgs/OLMo3-1B-stage1-baseline.py \
  --save-folder=/path/to/checkpoints/baseline \
  --work-dir=/path/to/work/baseline \
  --data-root=/path/to/tokenized-data \
  --name=olmo3-1b-baseline

torchrun --nproc-per-node=8 \
  cfgs/OLMo3-1B-stage1-situ.py \
  --save-folder=/path/to/checkpoints/situ \
  --work-dir=/path/to/work/situ \
  --data-root=/path/to/tokenized-data \
  --name=olmo3-1b-situ

torchrun --nproc-per-node=8 \
  cfgs/OLMo3-1B-stage1-asymmetric-rational-clip.py \
  --save-folder=/path/to/checkpoints/rational-clip \
  --work-dir=/path/to/work/rational-clip \
  --data-root=/path/to/tokenized-data \
  --name=olmo3-1b-asymmetric-rational-clip

torchrun --nproc-per-node=8 \
  cfgs/OLMo3-1B-stage1-mose-rmsnorm-silu.py \
  --save-folder=/path/to/checkpoints/mose-rmsnorm-silu \
  --work-dir=/path/to/work/mose-rmsnorm-silu \
  --data-root=/path/to/tokenized-data \
  --name=olmo3-1b-mose-rmsnorm-silu

torchrun --nproc-per-node=8 \
  cfgs/OLMo3-1B-stage1-mose-situ.py \
  --save-folder=/path/to/checkpoints/mose-situ \
  --work-dir=/path/to/work/mose-situ \
  --data-root=/path/to/tokenized-data \
  --name=olmo3-1b-mose-situ

torchrun --nproc-per-node=8 \
  cfgs/OLMo3-1B-stage1-mose-asymmetric-rational-clip.py \
  --save-folder=/path/to/checkpoints/mose-rational-clip \
  --work-dir=/path/to/work/mose-rational-clip \
  --data-root=/path/to/tokenized-data \
  --name=olmo3-1b-mose-asymmetric-rational-clip
```

Override ranks directly from either MoSE entry point. Setting both down ranks to zero selects the
dense down projection:

```bash
--model.block.feed_forward.r1=880 \
--model.block.feed_forward.r2=880 \
--model.block.feed_forward.down_r1=0 \
--model.block.feed_forward.down_r2=0
```

The nonlinear expert functions can be overridden independently from any MoSE entry point:

```bash
--model.block.feed_forward.gate_nonlinearity=rms_norm \
--model.block.feed_forward.up_nonlinearity=rms_norm \
--model.block.feed_forward.down_nonlinearity=silu
```

By default, MoSE starts at zero-based layer 0, so every transformer layer uses it. Set
`--mose-start-layer` on any MoSE entry point to leave the earlier layers dense while retaining the
same channel control (standard control uses native SwiGLU). For example, this keeps layers 0 and 1
dense and enables MoSE from layer 2 onward:

```bash
torchrun --nproc-per-node=8 \
  cfgs/OLMo3-1B-stage1-mose-asymmetric-rational-clip.py \
  --mose-start-layer=2 \
  --save-folder=/path/to/checkpoints/mose-from-layer-2 \
  --work-dir=/path/to/work/mose-from-layer-2 \
  --data-root=/path/to/tokenized-data \
  --name=olmo3-1b-mose-from-layer-2
```

`--data-root` must provide the layouts required by `OLMo_mix_0625_150Bsample` and
`v3_small_ppl_validation`. For an initial server smoke test without external logging or evaluation:

`--sequence-length` must be positive, divide `max(4096, sequence_length)`, and produce a
`16 * sequence_length` rank microbatch that divides the global token batch. The per-rank microbatch
token count scales with the selected sequence length.

```bash
--trainer.callbacks.wandb.enabled=false \
--trainer.callbacks.lm_evaluator.enabled=false \
--trainer.callbacks.downstream_evaluator.enabled=false
```
