#!/usr/bin/env bash
set -euo pipefail

# OLMES "base easy" evaluation for the converted SiTU-Up checkpoint.
#
# Throughput layout:
#   - 8 OLMES workers over 8 GPUs (DP=8, TP=1)
#   - fixed per-worker batch size 32
#
# The RJob cannot access the Internet, so every model, tokenizer, dataset and
# Python dependency below resolves from the mounted GPFS2 tree.

GPFS_ROOT="/mnt/shared-storage-gpfs2/intern-pretrain-shared02/shijiayang-p"
OLMES_REPO="${GPFS_ROOT}/archspace/olmes_official"
OLMES_ENV="${GPFS_ROOT}/conda_envs/olmes"
MODEL_DIR="${GPFS_ROOT}/ckpts/huggingface/olmo3-1b-situ-up-step75080"
OUTPUT_DIR="${OLMES_OUTPUT_DIR:-${GPFS_ROOT}/olmo3_outputs/evals/olmes/situ-up-step75080-base-easy-dp8-bs32-sharded}"
CACHE_ROOT="${GPFS_ROOT}/olmes_cache"
RUNTIME_HOOKS="${GPFS_ROOT}/archspace-MoSE/tools/olmes_runtime"
DP_DRIVER="${RUNTIME_HOOKS}/run_easy_dp_sharded.py"

NUM_GPUS="${OLMES_NUM_GPUS:-8}"
NUM_WORKERS="${OLMES_NUM_WORKERS:-8}"
BATCH_SIZE="${OLMES_BATCH_SIZE:-32}"

if [[ ! -x "${OLMES_ENV}/bin/python" ]]; then
    echo "Missing OLMES Python: ${OLMES_ENV}/bin/python" >&2
    exit 1
fi
if [[ ! -f "${OLMES_REPO}/oe_eval/launch.py" ]]; then
    echo "Missing OLMES repository: ${OLMES_REPO}" >&2
    exit 1
fi
if [[ ! -f "${MODEL_DIR}/model.safetensors" ]]; then
    echo "Missing converted model: ${MODEL_DIR}" >&2
    exit 1
fi
if [[ ! -f "${DP_DRIVER}" ]]; then
    echo "Missing OLMES DP driver: ${DP_DRIVER}" >&2
    exit 1
fi
if (( NUM_GPUS % NUM_WORKERS != 0 )); then
    echo "OLMES_NUM_GPUS (${NUM_GPUS}) must be divisible by OLMES_NUM_WORKERS (${NUM_WORKERS})." >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}" "${CACHE_ROOT}/tmp"

export PATH="${OLMES_ENV}/bin:${PATH}"
export PYTHONPATH="${RUNTIME_HOOKS}:${OLMES_REPO}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONNOUSERSITE=1

# Fully offline Hugging Face / datasets setup.
export HF_HOME="${CACHE_ROOT}/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export LITELLM_LOCAL_MODEL_COST_MAP=True
export WANDB_MODE=offline
export WANDB_DISABLED=true

export TMPDIR="${CACHE_ROOT}/tmp"
export TEMP="${TMPDIR}"
export TMP="${TMPDIR}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

# Do not let a proxy turn an accidental remote lookup into a long timeout.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

echo "OLMES repo:    ${OLMES_REPO}"
echo "OLMES Python:  ${OLMES_ENV}/bin/python"
echo "Model:         ${MODEL_DIR}"
echo "Output:        ${OUTPUT_DIR}"
echo "Parallelism:   DP=${NUM_WORKERS}, GPUs=${NUM_GPUS}, GPUs/worker=$((NUM_GPUS / NUM_WORKERS))"
echo "Batch/worker:  ${BATCH_SIZE}"
echo "HF offline:    ${HF_HUB_OFFLINE}"

cd "${OLMES_REPO}"
exec "${OLMES_ENV}/bin/python" "${DP_DRIVER}" \
    --olmes-repo "${OLMES_REPO}" \
    --model "${MODEL_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --num-shards "${NUM_WORKERS}" \
    --batch-size "${BATCH_SIZE}"
