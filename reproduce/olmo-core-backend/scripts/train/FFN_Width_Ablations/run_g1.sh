#!/usr/bin/env bash
set -eo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
export MASTER_PORT=${MASTER_PORT:-30213}
export PYTHONDONTWRITEBYTECODE=1
export MOSE_CHECKPOINT_ROOT="${MOSE_CHECKPOINT_ROOT:-/mnt/shared-storage-gpfs2/intern-pretrain-shared02/taoji/ckpts/Olmo3-MoSE}"
exec "$SCRIPT_DIR/_run_common.sh" g1 olmo3-1b-qk-adaptive-g1-c2-q0-k1-m8192-ug568-down520-down-rms-fp8-30b "$@"
