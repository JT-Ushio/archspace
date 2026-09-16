#!/usr/bin/env bash
set -eo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
source "${BACKEND_DIR}/scripts/prepare/prepare_olmo_gpfs2.sh"
set -u
source /mnt/shared-storage-user/jitao/.local/bin/hceph_start
cd "${MOSE_JT}"
INIT_ARTIFACT_DIR="${INIT_ARTIFACT_DIR:-${MOSE_JT}/scripts/analysis/Initialization_Analysis/results/svd-init-r512-20260913-v3}"
exec "${OLMO_ENV}/bin/python" -m torch.distributed.run \
  --nproc_per_node="${PROC_PER_NODE:-8}" --nnodes="${NODE_COUNT:-1}" \
  --node_rank="${NODE_RANK:-0}" --master_addr="${MASTER_ADDR:-127.0.0.1}" \
  --master_port="${MASTER_PORT:-29653}" \
  scripts/analysis/Initialization_Analysis/probe.py \
  --output-dir="${INIT_ARTIFACT_DIR}" --data-root="${DATA_DIR}" \
  --work-dir="${WORK_DIR}" --num-sequences=32 --tokens-per-sequence=128
