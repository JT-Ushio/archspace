#!/usr/bin/env bash
set -euo pipefail

BACKEND_DIR=/mnt/shared-storage-gpfs2/intern-pretrain-shared02/shijiayang-p/archspace-MoSE/reproduce/olmo-core-backend
CHECKPOINT=/mnt/shared-storage-gpfs2/intern-pretrain-shared02/shijiayang-p/ckpts/olmo3-1b-SiTU-up-5e-3/stage1/step75080
HF_OUTPUT=/mnt/shared-storage-gpfs2/intern-pretrain-shared02/shijiayang-p/ckpts/huggingface/olmo3-1b-situ-up-step75080
PYTHON_BIN=/mnt/shared-storage-user/shijiayang-p/anaconda3/envs/olmo_core/bin/python
TOKENIZER_DIR=${BACKEND_DIR}/assets/dolma2-tokenizer

export BRAIN_USERNAME=shijiayang-p
export TMPDIR=/mnt/shared-storage-gpfs2/intern-pretrain-shared02/shijiayang-p/tmp/hf-situ-up-convert
export TEMP="${TMPDIR}"
export TMP="${TMPDIR}"
export HF_HOME=/mnt/shared-storage-gpfs2/intern-pretrain-shared02/shijiayang-p/hf_home
export PYTHON_BIN
OLMO_CORE_SRC=/mnt/shared-storage-gpfs2/intern-pretrain-shared02/shijiayang-p/archspace/OLMo-core/src
export PYTHONPATH=${OLMO_CORE_SRC}:${BACKEND_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p "${TMPDIR}" "$(dirname -- "${HF_OUTPUT}")"

if [[ ! -r "${CHECKPOINT}/config.json" ]]; then
  echo "Checkpoint is not readable: ${CHECKPOINT}/config.json" >&2
  exit 1
fi
if [[ ! -d "${CHECKPOINT}/model_and_optim" ]]; then
  echo "Checkpoint model state is missing: ${CHECKPOINT}/model_and_optim" >&2
  exit 1
fi
if [[ ! -f "${OLMO_CORE_SRC}/olmo_core/__init__.py" ]]; then
  echo "OLMo Core source is missing from the mounted GPFS2 workspace" >&2
  exit 1
fi
if [[ ! -s "${TOKENIZER_DIR}/tokenizer.json" ]]; then
  echo "Local tokenizer is missing: ${TOKENIZER_DIR}/tokenizer.json" >&2
  exit 1
fi
REUSE_STANDARD_OUTPUT=0
if [[ -e "${HF_OUTPUT}" ]] && [[ -n "$(find "${HF_OUTPUT}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  if [[ -s "${HF_OUTPUT}/conversion_report.json" ]]; then
    echo "A completed HF output already exists: ${HF_OUTPUT}" >&2
    exit 1
  elif [[ -s "${HF_OUTPUT}/model.safetensors" ]] && [[ -s "${HF_OUTPUT}/config.json" ]]; then
    echo "Reusing complete standard HF weights already present at: ${HF_OUTPUT}"
    REUSE_STANDARD_OUTPUT=1
  else
    INCOMPLETE_OUTPUT=${HF_OUTPUT}.incomplete.$(date +%Y%m%d-%H%M%S)
    echo "Archiving incomplete output to: ${INCOMPLETE_OUTPUT}"
    mv -- "${HF_OUTPUT}" "${INCOMPLETE_OUTPUT}"
  fi
fi

echo "Converting: ${CHECKPOINT}"
echo "HF output: ${HF_OUTPUT}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

REUSE_ARGS=()
if [[ "${REUSE_STANDARD_OUTPUT}" -eq 1 ]]; then
  REUSE_ARGS+=(--reuse-standard-hf-output)
fi

exec bash "${BACKEND_DIR}/run_convert_SiTU_up_checkpoint_to_HF.sh" \
  "${CHECKPOINT}" \
  "${HF_OUTPUT}" \
  "${REUSE_ARGS[@]}" \
  --tokenizer "${TOKENIZER_DIR}" \
  --device cpu \
  --decode-device cuda:0 \
  --decode-max-new-tokens 48
