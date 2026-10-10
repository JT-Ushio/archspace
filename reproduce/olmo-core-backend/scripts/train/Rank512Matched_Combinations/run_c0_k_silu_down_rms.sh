#!/usr/bin/env bash
# Train-CE control: K-only SiLU, Gate/Up Identity and Down RMSNorm.
set -eo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
if [[ ! -v MASTER_PORT ]]; then MASTER_PORT=29700; fi
export MASTER_PORT
exec "$SCRIPT_DIR/_run_common.sh" \
  c0_k_silu_down_rms \
  olmo3-1b-r512matched-c0-k-silu-down-rms-fp8-30b \
  --model.block.sequence_mixer.rank=512 \
  --model.block.sequence_mixer.q_rank=512 \
  --model.block.sequence_mixer.k_rank=512 \
  --model.block.sequence_mixer.v_rank=512 \
  --model.block.sequence_mixer.o_rank=512 \
  --model.block.sequence_mixer.share_scope=none \
  --model.block.sequence_mixer.q_nonlinearity=identity \
  --model.block.sequence_mixer.k_nonlinearity=silu \
  --model.block.sequence_mixer.v_nonlinearity=identity \
  --model.block.sequence_mixer.o_nonlinearity=identity \
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
