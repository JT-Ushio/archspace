#!/usr/bin/env bash
set -eo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
export MASTER_PORT=${MASTER_PORT:-29820}
export PYTHONDONTWRITEBYTECODE=1
export MOSE_CHECKPOINT_ROOT=/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/ckpts/Olmo3-MoSE
exec "$SCRIPT_DIR/_run_common.sh" p0qk olmo3-1b-ug-partial-p0qk-s0-p512-linear-qk-silu-down-rms-fp8-30b "$@"
