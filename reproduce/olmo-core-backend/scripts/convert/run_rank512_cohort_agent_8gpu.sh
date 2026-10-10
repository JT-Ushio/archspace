#!/usr/bin/env bash
# Agent 1x8 allocation; CPU-only streaming conversion within that allocation.
set -eo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
TASK_USER_ROOT="${TASK_USER_ROOT:-/mnt/shared-storage-gpfs2/intern-pretrain-shared02/shijiayang-p}"
OLMO_ENV="${OLMO_ENV:-${TASK_USER_ROOT}/conda_envs/olmo}"
CORE_SRC="${CORE_SRC:-${TASK_USER_ROOT}/archspace/OLMo-core-muon-fix/src}"
CKPT_ROOT="/mnt/shared-storage-gpfs2/intern-pretrain-shared02/taoji/ckpts/Olmo3-MoSE"
OUTPUT_ROOT="${OUTPUT_ROOT:-${TASK_USER_ROOT}/ckpts/huggingface/rank512-c0-c8-step14306-20261009-r2}"
DEPS_DIR="${DEPS_DIR:-${TASK_USER_ROOT}/ckpts/huggingface/dense-single-silu-step14306-20261008/.hf-export-deps}"
export BRAIN_USERNAME=jitao PS1="${PS1:-}"
source "${OLMO_ENV}/bin/activate"
if [[ ! -d "${CKPT_ROOT}" ]]; then
    source /mnt/shared-storage-user/jitao/.local/bin/hceph_start
fi
set -u
if [[ ! -d "${DEPS_DIR}/transformers" ]]; then
    echo "Missing previously verified HF export dependencies: ${DEPS_DIR}" >&2
    exit 1
fi
mkdir -p "${OUTPUT_ROOT}"
exec > >(tee -a "${OUTPUT_ROOT}/conversion.log") 2>&1
RUNTIME_TMP="$(mktemp -d /tmp/mose-hf-rank512.XXXXXXXX)"
trap 'rm -rf -- "${RUNTIME_TMP}"' EXIT
export TMPDIR="${RUNTIME_TMP}" TEMP="${RUNTIME_TMP}" TMP="${RUNTIME_TMP}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="${DEPS_DIR}:${CORE_SRC}:${BACKEND_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
"${OLMO_ENV}/bin/python" -u "${SCRIPT_DIR}/export_manifest_cohort.py" \
    "${SCRIPT_DIR}/rank512_c0_c8_cohort.json" --output-root "${OUTPUT_ROOT}" \
    --tokenizer "${BACKEND_DIR}/assets/dolma2-tokenizer" --dtype auto
