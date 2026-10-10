#!/usr/bin/env bash
set -eo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
export MASTER_PORT="${MASTER_PORT:-29860}"
exec "$SCRIPT_DIR/_run_common.sh" l1 \
  olmo3-1b-ffn-residual-l1-m8192-r512-unshared-linear-down-rms-fp8-30b \
  "$@"
