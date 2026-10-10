#!/usr/bin/env bash
set -eo pipefail
MOSE_RESOLVE_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
MOSE_RESOLVE_BACKEND="$(cd -- "$MOSE_RESOLVE_DIR/../../.." && pwd)"
export PYTHONDONTWRITEBYTECODE=1
source "$MOSE_RESOLVE_BACKEND/scripts/prepare/prepare_olmo_gpfs2.sh"
source /mnt/shared-storage-user/jitao/.local/bin/hceph_start
set -u
[[ "${NODE_COUNT:-1}" == 1 && "${PROC_PER_NODE:-8}" == 8 ]] || exit 2
exec "$OLMO_ENV/bin/python" "$MOSE_RESOLVE_DIR/resolve_dense.py"
