#!/usr/bin/env bash
set -eo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
export MASTER_PORT=${MASTER_PORT:-29811}
export PYTHONDONTWRITEBYTECODE=1
export MOSE_CHECKPOINT_ROOT=/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/ckpts/Olmo3-MoSE
exec "$SCRIPT_DIR/_run_common.sh" p1 olmo3-1b-ug-partial-p1-s560-p0-linear-down-rms-fp8-30b \
  trainer.load_path=/mnt/shared-storage-gpfs2/intern-pretrain-shared02/taoji/ckpts/Olmo3-MoSE/olmo3-1b-ug-partial-p1-s560-p0-linear-down-rms-fp8-30b/stage1/step0 \
  trainer.load_strategy=always "$@"
