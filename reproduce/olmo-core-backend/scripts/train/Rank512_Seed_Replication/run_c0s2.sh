#!/usr/bin/env bash
set -eo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
export MASTER_PORT=${MASTER_PORT:-29840}
export PYTHONDONTWRITEBYTECODE=1
export MOSE_CHECKPOINT_ROOT=/mnt/shared-storage-gpfs2/intern-pretrain-shared02/taoji/ckpts/Olmo3-MoSE
exec "$SCRIPT_DIR/_run_common.sh" c0s2 olmo3-1b-r512matched-c0-seed12537-data34521-fp8-30b --init_seed=12537 --data_loader.seed=34521 "$@"
