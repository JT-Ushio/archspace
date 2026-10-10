#!/usr/bin/env bash
set -eo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
export MASTER_PORT=${MASTER_PORT:-29823}
export PYTHONDONTWRITEBYTECODE=1
export MOSE_CHECKPOINT_ROOT=/mnt/shared-storage-gpfs2/intern-pretrain-shared02/taoji/ckpts/Olmo3-MoSE
exec "$SCRIPT_DIR/_run_common.sh" p3qk olmo3-1b-ug-partial-p3qk-s320-p224-linear-qk-silu-down-rms-fp8-30b "$@"
