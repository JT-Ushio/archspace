#!/usr/bin/env bash
set -eo pipefail
VARIANT="${1:?Expected identity, static, dynamic or dynamic_residual}"
shift
case "${VARIANT}" in
  identity|static|dynamic|dynamic_residual) ;;
  *) echo "Unknown Down rotation variant: ${VARIANT}" >&2; exit 2 ;;
esac
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
RUN_VARIANT="${VARIANT//_/-}"
RUN_NAME="${RUN_NAME:-olmo3-1b-down-output-subspace-${RUN_VARIANT}-g32-a05-fp8-30b}"
STAGE="${STAGE:-stage1}"
PROC_PER_NODE="${PROC_PER_NODE:-8}"
NODE_COUNT="${NODE_COUNT:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29641}"
export WANDB_API_KEY="${WANDB_SJY}"
# Hyperparameters/architecture live in the shared Python builder, inherited
# directly from the 30B Down-Identity baseline; avoid conflicting shell defaults.
exec "${OLMO_ENV}/bin/python" -m torch.distributed.run \
  --nproc_per_node="${PROC_PER_NODE}" \
  --nnodes="${NODE_COUNT}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  "cfps/BS256_Seq8K_30B_WSD_FP8/Down_Output_Subspace_Ablations/down_${VARIANT}_fp8.py" \
  --name "${RUN_NAME}" \
  --sequence-length=8192 \
  --data-root="${DATA_DIR}" \
  --save-folder="${CKPT_DIR}/${RUN_NAME}/${STAGE}" \
  --work-dir "${WORK_DIR}" \
  trainer.work_dir="${WORK_DIR}" \
  "$@"
