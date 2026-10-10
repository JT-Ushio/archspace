#!/usr/bin/env bash
set -eo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
export MASTER_PORT=${MASTER_PORT:-29812}
export PYTHONDONTWRITEBYTECODE=1
export MOSE_STAGE_OVERRIDE=stage1-resume7152
export MOSE_CHECKPOINT_ROOT=/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/ckpts/Olmo3-MoSE
exec "$SCRIPT_DIR/_run_common.sh" p2 olmo3-1b-ug-partial-p2-s480-p80-linear-down-rms-fp8-30b \
  trainer.load_path=/mnt/shared-storage-gpfs2/intern-pretrain-shared02/taoji/ckpts/Olmo3-MoSE/olmo3-1b-ug-partial-p2-s480-p80-linear-down-rms-fp8-30b/stage1/step7152 \
  trainer.load_strategy=always "$@"
