#!/usr/bin/env bash
# Allocate agent 1x8. Conversion itself is CPU-only and uses one process.
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
TASK_USER_ROOT="${TASK_USER_ROOT:-/mnt/shared-storage-gpfs2/intern-pretrain-shared02/shijiayang-p}"
OLMO_ENV="${OLMO_ENV:-${TASK_USER_ROOT}/conda_envs/olmo}"
CORE_SRC="${CORE_SRC:-${TASK_USER_ROOT}/archspace/OLMo-core-muon-fix/src}"
CKPT_ROOT="${CKPT_ROOT:-/mnt/shared-storage-gpfs2/intern-pretrain-shared02/taoji/ckpts/Olmo3-MoSE}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${TASK_USER_ROOT}/ckpts/huggingface/dense-single-silu-step14306-20261008}"
DEPS_DIR="${DEPS_DIR:-${OUTPUT_ROOT}/.hf-export-deps}"
export BRAIN_USERNAME=jitao
export PS1="${PS1:-}"
source "${OLMO_ENV}/bin/activate"
if [[ ! -d "${CKPT_ROOT}" ]]; then
  HCEPH_START=/mnt/shared-storage-user/jitao/.local/bin/hceph_start
  if [[ ! -f "${HCEPH_START}" ]]; then
    echo "Checkpoint root and mount helper are unavailable: ${CKPT_ROOT}" >&2
    exit 1
  fi
  source "${HCEPH_START}"
fi
set -u
mkdir -p "${OUTPUT_ROOT}" "${OUTPUT_ROOT}/tmp"
exec > >(tee -a "${OUTPUT_ROOT}/conversion.log") 2>&1
echo "Exporting Dense + Q/K/Up/Gate single-SiLU final checkpoints to ${OUTPUT_ROOT}"
export TMPDIR="${OUTPUT_ROOT}/tmp"
export TEMP="${TMPDIR}"
export TMP="${TMPDIR}"
export HF_HOME="${OUTPUT_ROOT}/hf-cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
# Avoid depending on FlashAttention libraries during meta construction.
export CUDA_VISIBLE_DEVICES=""
PYTHON_BIN="${OLMO_ENV}/bin/python"
"${PYTHON_BIN}" -m pip install --disable-pip-version-check --no-deps --target "${DEPS_DIR}" \
  'transformers==4.57.6' 'huggingface-hub==0.36.0' 'regex>=2025.9.18' 'tokenizers==0.22.2' 'safetensors==0.8.0'
export PYTHONPATH="${DEPS_DIR}:${CORE_SRC}:${BACKEND_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${BACKEND_DIR}"
exec "${PYTHON_BIN}" -u "${SCRIPT_DIR}/export_dense_single_silu.py" \
  --checkpoint-root "${CKPT_ROOT}" --output-root "${OUTPUT_ROOT}" \
  --tokenizer "${BACKEND_DIR}/assets/dolma2-tokenizer" \
  --step 14306 --dtype auto --max-shard-size 1GB
