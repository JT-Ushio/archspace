#!/usr/bin/env bash
set -euo pipefail

BACKEND_DIR=/mnt/shared-storage-gpfs2/intern-pretrain-shared02/shijiayang-p/archspace-MoSE/reproduce/olmo-core-backend
HF_ROOT=/mnt/shared-storage-gpfs2/intern-pretrain-shared02/shijiayang-p/ckpts/huggingface
HF_OUTPUT=${HF_ROOT}/olmo3-1b-situ-up-step75080
OLD_DUPLICATE=${HF_OUTPUT}.incomplete.20260831-013559
REUSABLE_OUTPUT=${HF_OUTPUT}.incomplete.20260831-013844

if [[ ! -s "${REUSABLE_OUTPUT}/model.safetensors" ]]; then
  echo "Reusable converted weights are missing: ${REUSABLE_OUTPUT}/model.safetensors" >&2
  exit 1
fi

# These are only failed products from this conversion attempt. The native
# checkpoint lives in a different directory and is never touched here.
rm -rf -- "${HF_OUTPUT}" "${OLD_DUPLICATE}"
mv -- "${REUSABLE_OUTPUT}" "${HF_OUTPUT}"

exec bash "${BACKEND_DIR}/run_convert_SiTU_up_step75080_to_HF_rjob.sh"
