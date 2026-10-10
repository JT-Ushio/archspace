#!/usr/bin/env bash
set -eo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
export MASTER_PORT=${MASTER_PORT:-30215}
export PYTHONDONTWRITEBYTECODE=1
export MOSE_CHECKPOINT_ROOT="${MOSE_CHECKPOINT_ROOT:-/mnt/shared-storage-gpfs2/intern-pretrain-shared02/taoji/ckpts/Olmo3-MoSE}"
exec "$SCRIPT_DIR/_run_common.sh" w1m olmo3-1b-ffn-width-w1m-m6144-ug568-down520-c2-k-silu-down-rms-fp8-30b "$@"
