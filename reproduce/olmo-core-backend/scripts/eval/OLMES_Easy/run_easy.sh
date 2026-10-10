#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="/mnt/shared-storage-gpfs2/intern-pretrain-shared02/shijiayang-p"
if [[ ! -d "${DEFAULT_ROOT}/conda_envs/olmes" ]]; then
    DEFAULT_ROOT="/mnt/shared-storage-user/intern-pretrain-shared02/shijiayang-p"
fi
GPFS_ROOT="${OLMES_GPFS_ROOT:-${DEFAULT_ROOT}}"
OLMES_REPO="${OLMES_REPO:-${GPFS_ROOT}/archspace/olmes_official}"
OLMES_ENV="${OLMES_ENV:-${GPFS_ROOT}/conda_envs/olmes}"
CACHE_ROOT="${OLMES_CACHE_ROOT:-${GPFS_ROOT}/olmes_cache}"
MODEL_DIR="${OLMES_MODEL_DIR:-}"
if (( $# > 0 )) && [[ "$1" != --* ]]; then
    MODEL_DIR="$1"
    shift
fi
if [[ -z "${MODEL_DIR}" ]]; then
    echo "Usage: run_easy.sh /absolute/HF/model/directory [driver options]" >&2
    exit 2
fi
NUM_GPUS="${OLMES_NUM_GPUS:-8}"
NUM_WORKERS="${OLMES_NUM_WORKERS:-8}"
BATCH_SIZE="${OLMES_BATCH_SIZE:-32}"
if [[ "${NUM_GPUS}" != 8 || "${NUM_WORKERS}" != 8 ]]; then
    echo "This launcher uses one 8-GPU node, DP8 / TP1 (one model per GPU)." >&2
    exit 2
fi
if [[ ! -x "${OLMES_ENV}/bin/python" || ! -f "${OLMES_REPO}/oe_eval/launch.py" ]]; then
    echo "Missing OLMES environment or repository: ${OLMES_ENV}; ${OLMES_REPO}" >&2
    exit 1
fi
OUTPUT_DIR="${OLMES_OUTPUT_DIR:-${GPFS_ROOT}/olmo3_outputs/evals/olmes/$(basename -- "${MODEL_DIR}")-easy-balanced-bs${BATCH_SIZE}-$(date +%Y%m%d-%H%M%S)-$$}"
RUNTIME_TMP="$(mktemp -d /tmp/mose-olmes-easy.XXXXXXXX)"
trap 'rm -rf -- "${RUNTIME_TMP}"' EXIT
export TMPDIR="${RUNTIME_TMP}"
export TEMP="${TMPDIR}" TMP="${TMPDIR}"
export PATH="${OLMES_ENV}/bin:${PATH}"
export PYTHONPATH="${SCRIPT_DIR}:${OLMES_REPO}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONNOUSERSITE=1
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
export HF_HOME="${CACHE_ROOT}/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export LITELLM_LOCAL_MODEL_COST_MAP=True
export WANDB_MODE=offline WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
echo "OLMES: ${OLMES_REPO}; Python: ${OLMES_ENV}/bin/python"
echo "Model: ${MODEL_DIR}; Output: ${OUTPUT_DIR}"
echo "DP8 / TP1; requested batch per GPU: ${BATCH_SIZE}; wall-time balanced tasks"
"${OLMES_ENV}/bin/python" "${SCRIPT_DIR}/run_easy_dp_sharded.py" \
    --olmes-repo "${OLMES_REPO}" \
    --model "${MODEL_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --num-shards "${NUM_WORKERS}" \
    --batch-size "${BATCH_SIZE}" \
    "$@"
