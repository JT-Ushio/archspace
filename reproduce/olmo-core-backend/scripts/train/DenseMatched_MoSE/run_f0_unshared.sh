#!/usr/bin/env bash
# Dense-parameter-matched bottleneck control: no sharing; Down uses RMSNorm.
set -eo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
if [[ ! -v MASTER_PORT ]]; then MASTER_PORT=29680; fi
export MASTER_PORT
exec "$SCRIPT_DIR/_run_common.sh" \
  f0_unshared \
  olmo3-1b-densematched-mose-f0-unshared-down-rms-fp8-30b \
  --model.block.sequence_mixer.rank=1024 \
  --model.block.sequence_mixer.q_rank=1024 \
  --model.block.sequence_mixer.k_rank=1024 \
  --model.block.sequence_mixer.v_rank=1024 \
  --model.block.sequence_mixer.o_rank=1024 \
  --model.block.sequence_mixer.share_scope=none \
  --model.block.sequence_mixer.q_nonlinearity=identity \
  --model.block.sequence_mixer.k_nonlinearity=identity \
  --model.block.sequence_mixer.v_nonlinearity=identity \
  --model.block.sequence_mixer.o_nonlinearity=identity \
  --model.block.sequence_mixer.rms_norm_learnable_weight=false \
  --model.block.feed_forward.r1=0 \
  --model.block.feed_forward.r2=1640 \
  --model.block.feed_forward.down_r1=0 \
  --model.block.feed_forward.down_r2=1632 \
  --model.block.feed_forward.share_gate_up_subspace=false \
  --model.block.feed_forward.control=standard \
  --model.block.feed_forward.control_scope=none \
  --model.block.feed_forward.gate_nonlinearity=silu \
  --model.block.feed_forward.up_nonlinearity=silu \
  --model.block.feed_forward.down_nonlinearity=rms_norm \
  --model.block.feed_forward.rms_norm_learnable_weight=false \
  "$@"
