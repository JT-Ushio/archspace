#!/usr/bin/env bash
set -eo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
export MASTER_PORT=${MASTER_PORT:-30211}
export PYTHONDONTWRITEBYTECODE=1
export MOSE_CHECKPOINT_ROOT="${MOSE_CHECKPOINT_ROOT:-/mnt/shared-storage-gpfs2/intern-pretrain-shared02/taoji/ckpts/Olmo3-MoSE}"
exec "$SCRIPT_DIR/_run_common.sh" w1 olmo3-1b-ffn-width-w1-m6144-ug736-down640-c2-k-silu-down-rms-fp8-30b "$@"
