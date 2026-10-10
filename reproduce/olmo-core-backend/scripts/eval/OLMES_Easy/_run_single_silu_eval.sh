#!/usr/bin/env bash
set -euo pipefail
VARIANT="${1:?Pass dense, q, k, up, or gate}"
shift
case "${VARIANT}" in
    dense|q|k|up|gate) ;;
    *) echo "Unknown single-SiLU evaluation model: ${VARIANT}" >&2; exit 2 ;;
esac
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
GPFS_ROOT="${OLMES_GPFS_ROOT:-/mnt/shared-storage-gpfs2/intern-pretrain-shared02/shijiayang-p}"
export OLMES_GPFS_ROOT="${GPFS_ROOT}"
export OLMES_MODEL_DIR="${GPFS_ROOT}/ckpts/huggingface/dense-single-silu-step14306-20261008/${VARIANT}-step14306"
export OLMES_OUTPUT_DIR="${OLMES_OUTPUT_DIR:-${GPFS_ROOT}/olmo3_outputs/evals/olmes/dense-single-silu-step14306-20261008-gpu-20261008-2124/${VARIANT}}"
export OLMES_NUM_GPUS=8 OLMES_NUM_WORKERS=8
export OLMES_BATCH_SIZE="${OLMES_BATCH_SIZE:-32}"
exec bash "${SCRIPT_DIR}/run_easy.sh" "$@"
