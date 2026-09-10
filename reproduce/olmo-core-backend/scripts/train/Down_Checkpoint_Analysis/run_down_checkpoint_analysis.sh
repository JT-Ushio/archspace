#!/usr/bin/env bash
# Submit ONE node / ONE GPU. Four checkpoints are processed sequentially.
# This is analysis only: no optimizer, no training, no W&B writes.
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

# Defaults follow the saved configs of the four completed 30B Muon runs.
# Override *_CKPT with exact step dirs, or CKPT_ROOT with a remapped mount.
CKPT_ROOT="${CKPT_ROOT:-/mnt/shared-storage-gpfs2/intern-pretrain-shared02/taoji/ckpts/Olmo3-MoSE}"
S_CKPT="${S_CKPT:-${CKPT_ROOT}/olmo3-1b-down-activation-silu-fp8-30b/stage1}"
I_CKPT="${I_CKPT:-${CKPT_ROOT}/olmo3-1b-down-activation-identity-fp8-30b/stage1}"
R_CKPT="${R_CKPT:-${CKPT_ROOT}/olmo3-1b-down-activation-rms-fp8-30b/stage1}"
SR_CKPT="${SR_CKPT:-${CKPT_ROOT}/olmo3-1b-down-activation-silu-rms-norm-fp8-30b/stage1}"
ANALYSIS_NAME="${ANALYSIS_NAME:-down-ckpt-analysis-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${MOSE_JT}/scripts/analysis/Down_Checkpoint_Analysis/results/${ANALYSIS_NAME}}"
ANALYSIS_WORK_DIR="${ANALYSIS_WORK_DIR:-${WORK_DIR}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

exec "${OLMO_ENV}/bin/python" -u scripts/analysis/Down_Checkpoint_Analysis/analyze_checkpoints.py \
  --checkpoint "S=${S_CKPT}" \
  --checkpoint "I=${I_CKPT}" \
  --checkpoint "R=${R_CKPT}" \
  --checkpoint "SR=${SR_CKPT}" \
  --expected-step="${EXPECTED_STEP:-14306}" \
  --data-root="${DATA_DIR}" \
  --work-dir="${ANALYSIS_WORK_DIR}" \
  --output-dir="${OUTPUT_DIR}" \
  --num-sequences="${NUM_SEQUENCES:-128}" \
  --tokens-per-sequence="${TOKENS_PER_SEQUENCE:-128}" \
  --seed="${ANALYSIS_SEED:-34521}" \
  --cpu-threads="${ANALYSIS_CPU_THREADS:-8}" \
  --heatmap-sequences="${HEATMAP_SEQUENCES:-0}" \
  --heatmap-tokens="${HEATMAP_TOKENS:-64}" \
  --heatmap-channels="${HEATMAP_CHANNELS:-128}"
