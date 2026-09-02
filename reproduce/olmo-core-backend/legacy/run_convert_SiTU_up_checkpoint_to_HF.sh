#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 CHECKPOINT_STEP_DIR HF_OUTPUT_DIR [converter options]" >&2
  echo "Example: $0 \"\${CKPT_DIR}/olmo3-1b-SiTU-up-5e-3/stage1/step75080\" \"\${CKPT_DIR}/huggingface/olmo3-1b-situ-up-step75080\"" >&2
  exit 2
fi

CHECKPOINT=$1
HF_OUTPUT=$2
shift 2

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PYTHON_BIN=${PYTHON_BIN:-/mnt/shared-storage-user/jitao/envs/MoSE/bin/python}

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python environment not found or not executable: ${PYTHON_BIN}" >&2
  echo "Set PYTHON_BIN to the Python executable of the MoSE training environment." >&2
  exit 1
fi

for temp_name in TMPDIR TEMP TMP; do
  temp_path=${!temp_name:-}
  if [[ -n "${temp_path}" ]]; then
    mkdir -p "${temp_path}"
  fi
done

export PYTHONPATH="${SCRIPT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

exec "${PYTHON_BIN}" \
  "${SCRIPT_DIR}/tools/convert_situ_up_checkpoint_to_hf.py" \
  --checkpoint "${CHECKPOINT}" \
  --output "${HF_OUTPUT}" \
  "$@"
