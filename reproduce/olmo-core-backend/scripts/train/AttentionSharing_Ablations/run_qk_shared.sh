#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
source "${BACKEND_DIR}/scripts/prepare/prepare_olmo_gpfs2.sh"
set -u

HCEPH_START=/mnt/shared-storage-user/jitao/.local/bin/hceph_start
if [[ ! -f "${HCEPH_START}" ]]; then
  echo "Missing hceph mount helper: ${HCEPH_START}" >&2
  exit 1
fi
source "${HCEPH_START}"

cd "${MOSE_JT}"

RUN_NAME="${RUN_NAME:-olmo3-1b-qk-shared}"
STAGE="${STAGE:-stage1}"
PROC_PER_NODE="${PROC_PER_NODE:-8}"
NODE_COUNT="${NODE_COUNT:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29642}"

export WANDB_API_KEY="${WANDB_SJY}"

exec "${OLMO_ENV}/bin/python" -m torch.distributed.run \
  --nproc_per_node="${PROC_PER_NODE}" \
  --nnodes="${NODE_COUNT}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  cfps/BS256_Seq8K_30B_WSD_FP8/AttentionSharing_Ablations/qk_shared.py \
  --name "${RUN_NAME}" \
  --sequence-length=8192 \
  --data-root="${DATA_DIR}" \
  --save-folder="${CKPT_DIR}/${RUN_NAME}/${STAGE}" \
  --work-dir "${WORK_DIR}" \
  --data_loader.global_batch_size=2097152 \
  --train_module.rank_microbatch_size=65536 \
  --train_module.optim.lr=0.005 \
  --train_module.scheduler.warmup=2000 \
  --train_module.scheduler.decay_fraction=0.2 \
  --train_module.float8_config.enabled=true \
  --trainer.max_duration.value=30000000000 \
  --trainer.max_duration.unit=tokens \
  --trainer.callbacks.checkpointer.save_interval=3576 \
  --trainer.callbacks.checkpointer.max_checkpoints=null \
  --model.block.sequence_mixer.rank=512 \
  --model.block.sequence_mixer.share_scope=qk \
  --model.block.sequence_mixer.q_nonlinearity=silu \
  --model.block.sequence_mixer.k_nonlinearity=silu \
  --model.block.sequence_mixer.v_nonlinearity=silu \
  --model.block.sequence_mixer.o_nonlinearity=silu \
  --model.block.sequence_mixer.rms_norm_learnable_weight=false \
  --model.block.feed_forward.r1=0 \
  --model.block.feed_forward.r2=512 \
  --model.block.feed_forward.down_r1=0 \
  --model.block.feed_forward.down_r2=512 \
  --model.block.feed_forward.share_gate_up_subspace=true \
  --model.block.feed_forward.control=standard \
  --model.block.feed_forward.control_scope=none \
  --model.block.feed_forward.gate_nonlinearity=silu \
  --model.block.feed_forward.up_nonlinearity=silu \
  --model.block.feed_forward.down_nonlinearity=rms_norm \
  --model.block.feed_forward.rms_norm_learnable_weight=false \
  trainer.work_dir="${WORK_DIR}"
