#!/usr/bin/env bash
# Train-CE exploration: all Attention SiLU + FFN Identity/Down RMSNorm + K/V sharing.
set -eo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
if [[ ! -v MASTER_PORT ]]; then MASTER_PORT=29704; fi
export MASTER_PORT
exec "$SCRIPT_DIR/_run_common.sh" \
  c4_attention_silu_down_rms_kv \
  olmo3-1b-r512matched-c4-attention-silu-down-rms-kv-fp8-30b \
  --model.block.sequence_mixer.rank=512 \
  --model.block.sequence_mixer.q_rank=512 \
  --model.block.sequence_mixer.k_rank=688 \
  --model.block.sequence_mixer.v_rank=688 \
  --model.block.sequence_mixer.o_rank=504 \
  --model.block.sequence_mixer.share_scope=kv \
  --model.block.sequence_mixer.q_nonlinearity=silu \
  --model.block.sequence_mixer.k_nonlinearity=silu \
  --model.block.sequence_mixer.v_nonlinearity=silu \
  --model.block.sequence_mixer.o_nonlinearity=silu \
  --model.block.feed_forward.r1=0 \
  --model.block.feed_forward.r2=512 \
  --model.block.feed_forward.down_r1=0 \
  --model.block.feed_forward.down_r2=512 \
  --model.block.feed_forward.share_gate_up_subspace=false \
  --model.block.feed_forward.gate_nonlinearity=identity \
  --model.block.feed_forward.up_nonlinearity=identity \
  --model.block.feed_forward.down_nonlinearity=rms_norm \
  --model.block.feed_forward.rms_norm_learnable_weight=false \
  "$@"
