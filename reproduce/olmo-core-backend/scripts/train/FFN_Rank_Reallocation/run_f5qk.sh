#!/usr/bin/env bash
set -eo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
export MASTER_PORT=${MASTER_PORT:-29835}
export PYTHONDONTWRITEBYTECODE=1
export MOSE_CHECKPOINT_ROOT=/mnt/shared-storage-gpfs2/intern-pretrain-shared02/taoji/ckpts/Olmo3-MoSE
exec "$SCRIPT_DIR/_run_common.sh" f5qk olmo3-1b-ffn-budget-f5qk-ug560-down528-shared-qk-silu-ug-linear-down-rms-fp8-30b "$@"
