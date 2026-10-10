#!/usr/bin/env bash
# Combined candidate: K-only SiLU + Down RMSNorm + K/V and Gate/Up sharing.
set -eo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
if [[ ! -v MASTER_PORT ]]; then MASTER_PORT=29703; fi
export MASTER_PORT
exec "$SCRIPT_DIR/_run_common.sh" \
  c3_k_silu_down_rms_kv_gateup \
  olmo3-1b-r512matched-c3-k-silu-down-rms-kv-gateup-fp8-30b \
  --model.block.sequence_mixer.rank=520 \
  --model.block.sequence_mixer.q_rank=520 \
  --model.block.sequence_mixer.k_rank=680 \
  --model.block.sequence_mixer.v_rank=680 \
  --model.block.sequence_mixer.o_rank=512 \
  --model.block.sequence_mixer.share_scope=kv \
  --model.block.sequence_mixer.q_nonlinearity=identity \
  --model.block.sequence_mixer.k_nonlinearity=silu \
  --model.block.sequence_mixer.v_nonlinearity=identity \
  --model.block.sequence_mixer.o_nonlinearity=identity \
  --model.block.feed_forward.r1=0 \
  --model.block.feed_forward.r2=568 \
  --model.block.feed_forward.down_r1=0 \
  --model.block.feed_forward.down_r2=512 \
  --model.block.feed_forward.share_gate_up_subspace=true \
  --model.block.feed_forward.gate_nonlinearity=identity \
  --model.block.feed_forward.up_nonlinearity=identity \
  --model.block.feed_forward.down_nonlinearity=rms_norm \
  --model.block.feed_forward.rms_norm_learnable_weight=false \
  "$@"
