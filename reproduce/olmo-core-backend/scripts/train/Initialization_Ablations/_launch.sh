#!/usr/bin/env bash
set -eo pipefail
INIT_MODE="${1:?independent or svd_scale_matched required}"
case "${INIT_MODE}" in independent|svd_scale_matched) ;; *) exit 2 ;; esac
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
source "${BACKEND_DIR}/scripts/prepare/prepare_olmo_gpfs2.sh"
set -u
source /mnt/shared-storage-user/jitao/.local/bin/hceph_start
cd "${MOSE_JT}"
MODE_NAME="${INIT_MODE//_/-}"
RUN_NAME="${RUN_NAME:-olmo3-1b-init-${MODE_NAME}-down-rms-fp8-30b}"
INIT_ARTIFACT_DIR="${INIT_ARTIFACT_DIR:-${MOSE_JT}/scripts/analysis/Initialization_Analysis/results/svd-init-r512-20260913-v3}"
PROC_PER_NODE="${PROC_PER_NODE:-8}"
NODE_COUNT="${NODE_COUNT:-2}"
if (( 256 % (NODE_COUNT * PROC_PER_NODE * 4) != 0 )); then
  echo "BS256 incompatible with requested world size and microbatch4" >&2
  exit 1
fi
export WANDB_API_KEY="${WANDB_SJY}"
export WANDB_RESUME=allow
exec "${OLMO_ENV}/bin/python" -m torch.distributed.run \
  --nproc_per_node="${PROC_PER_NODE}" --nnodes="${NODE_COUNT}" \
  --node_rank="${NODE_RANK:-0}" --master_addr="${MASTER_ADDR:?}" \
  --master_port="${MASTER_PORT:-29652}" \
  "cfps/BS256_Seq8K_30B_WSD_FP8/Initialization_Ablations/${INIT_MODE}.py" \
  --name "${RUN_NAME}" --sequence-length=8192 \
  --data-root="${DATA_DIR}" --save-folder="${CKPT_DIR}/${RUN_NAME}/stage1" \
  --work-dir="${WORK_DIR}" \
  --model.factor_init="${INIT_MODE}" --model.factor_init_artifacts="${INIT_ARTIFACT_DIR}" \
  --data_loader.global_batch_size=2097152 --train_module.rank_microbatch_size=32768 \
  --train_module.optim.lr=0.005 --train_module.float8_config.enabled=true \
  --train_module.scheduler.warmup=2000 --train_module.scheduler.decay_fraction=0.2 \
  --trainer.max_duration.value=30000000000 --trainer.max_duration.unit=tokens \
  --trainer.callbacks.checkpointer.save_interval=1788 \
  --trainer.callbacks.checkpointer.max_checkpoints=null \
  trainer.work_dir="${WORK_DIR}"
