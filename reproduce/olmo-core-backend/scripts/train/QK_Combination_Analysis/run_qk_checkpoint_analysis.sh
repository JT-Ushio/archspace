#!/usr/bin/env bash
# 2x8 independent GPU workers, fixed held-out samples, read-only checkpoints.
set -eo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
source "${BACKEND_DIR}/scripts/prepare/prepare_olmo_gpfs2.sh"
set -u
HCEPH_START=/mnt/shared-storage-user/jitao/.local/bin/hceph_start
if [[ ! -f "${HCEPH_START}" ]]; then
  echo "Missing hceph helper: ${HCEPH_START}" >&2
  exit 1
fi
source "${HCEPH_START}"
cd "${MOSE_JT}"
CKPT_ROOT="${CKPT_ROOT:-/mnt/shared-storage-gpfs2/intern-pretrain-shared02/taoji/ckpts/Olmo3-MoSE}"
L_CKPT="${L_CKPT:-${CKPT_ROOT}/olmo3-1b-all-lowrank-linear-fp8-30b/stage1}"
Q_CKPT="${Q_CKPT:-${CKPT_ROOT}/olmo3-1b-all-linear-q-only-silu-fp8-30b/stage1}"
K_CKPT="${K_CKPT:-${CKPT_ROOT}/olmo3-1b-all-linear-k-only-silu-fp8-30b/stage1}"
# A scheduler-provided JOB_ID is shared by both nodes. Never use per-node time.
ANALYSIS_NAME="${ANALYSIS_NAME:-${JOB_ID:?JOB_ID or explicit ANALYSIS_NAME required}}"
OUTPUT_DIR="${OUTPUT_DIR:-${MOSE_JT}/scripts/analysis/QK_Combination_Analysis/results/${ANALYSIS_NAME}}"
export OMP_NUM_THREADS=2
exec "${OLMO_ENV}/bin/python" -m torch.distributed.run \
  --nproc_per_node="${PROC_PER_NODE:-8}" \
  --nnodes="${NODE_COUNT:-2}" \
  --node_rank="${NODE_RANK:-0}" \
  --master_addr="${MASTER_ADDR:?MASTER_ADDR required}" \
  --master_port="${MASTER_PORT:-29651}" \
  scripts/analysis/QK_Combination_Analysis/analyze.py \
  --checkpoint "L=${L_CKPT}" --checkpoint "Q=${Q_CKPT}" --checkpoint "K=${K_CKPT}" \
  --steps 3576 7152 10728 14306 \
  --output-dir="${OUTPUT_DIR}" --work-dir="${WORK_DIR}" --data-root="${DATA_DIR}" \
  --num-sequences="${NUM_SEQUENCES:-128}" \
  --tokens-per-sequence="${TOKENS_PER_SEQUENCE:-128}" \
  --queries="${QUERIES:-32}" --seed="${ANALYSIS_SEED:-34521}"
