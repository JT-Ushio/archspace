#!/usr/bin/env bash
# Q-only SiLU + K/V/O Identity + shared Gate/Up + Down RMSNorm.
set -eo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
if [[ ! -v MASTER_PORT ]]; then MASTER_PORT=29705; fi
export MASTER_PORT
exec "$SCRIPT_DIR/_run_common.sh" \
  c5_q_silu_down_rms_gateup \
  olmo3-1b-r512matched-c5-q-silu-down-rms-gateup-fp8-30b \
  --model.block.sequence_mixer.rank=512 \
  --model.block.sequence_mixer.q_rank=512 \
  --model.block.sequence_mixer.k_rank=512 \
  --model.block.sequence_mixer.v_rank=512 \
  --model.block.sequence_mixer.o_rank=496 \
  --model.block.sequence_mixer.share_scope=none \
  --model.block.sequence_mixer.q_nonlinearity=silu \
  --model.block.sequence_mixer.k_nonlinearity=identity \
  --model.block.sequence_mixer.v_nonlinearity=identity \
  --model.block.sequence_mixer.o_nonlinearity=identity \
  --model.block.feed_forward.r1=0 \
  --model.block.feed_forward.r2=568 \
  --model.block.feed_forward.down_r1=0 \
  --model.block.feed_forward.down_r2=520 \
  --model.block.feed_forward.share_gate_up_subspace=true \
  --model.block.feed_forward.gate_nonlinearity=identity \
  --model.block.feed_forward.up_nonlinearity=identity \
  --model.block.feed_forward.down_nonlinearity=rms_norm \
  --model.block.feed_forward.rms_norm_learnable_weight=false \
  "$@"
