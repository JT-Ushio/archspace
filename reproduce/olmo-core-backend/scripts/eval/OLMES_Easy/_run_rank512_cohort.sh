#!/usr/bin/env bash
set -euo pipefail
VARIANT="${1:?Pass a label from rank512_c0_c8_cohort.json}"
shift
case "${VARIANT}" in
    cola|full_linear|rms_down|q_only|k_only|c[0-8]) ;;
    *) echo "Unknown rank512 cohort variant: ${VARIANT}" >&2; exit 2 ;;
esac
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
GPFS_ROOT="${OLMES_GPFS_ROOT:-/mnt/shared-storage-gpfs2/intern-pretrain-shared02/shijiayang-p}"
export OLMES_GPFS_ROOT="${GPFS_ROOT}"
export OLMES_MODEL_DIR="${GPFS_ROOT}/ckpts/huggingface/rank512-c0-c8-step14306-20261009-r2/${VARIANT}-step14306"
export OLMES_OUTPUT_DIR="${OLMES_OUTPUT_DIR:-${GPFS_ROOT}/olmo3_outputs/evals/olmes/rank512-c0-c8-step14306-20261009/${VARIANT}}"
export OLMES_NUM_GPUS=8 OLMES_NUM_WORKERS=8 OLMES_BATCH_SIZE=32
exec bash "${SCRIPT_DIR}/run_easy.sh" "$@"
