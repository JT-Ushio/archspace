#!/usr/bin/env bash
set -eo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
export MASTER_PORT=${MASTER_PORT:-30214}
export PYTHONDONTWRITEBYTECODE=1
export MOSE_CHECKPOINT_ROOT="${MOSE_CHECKPOINT_ROOT:-/mnt/shared-storage-gpfs2/intern-pretrain-shared02/taoji/ckpts/Olmo3-MoSE}"
exec "$SCRIPT_DIR/_run_common.sh" w1qk olmo3-1b-ffn-width-w1qk-m6144-ug736-down640-c6-qk-silu-down-rms-fp8-30b "$@"
