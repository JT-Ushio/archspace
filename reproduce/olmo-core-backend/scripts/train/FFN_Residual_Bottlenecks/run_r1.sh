#!/usr/bin/env bash
set -eo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
export MASTER_PORT="${MASTER_PORT:-29862}"
exec "$SCRIPT_DIR/_run_common.sh" r1 \
  olmo3-1b-ffn-residual-r1-m2048-r1280-unshared-linear-down-rms-fp8-30b \
  "$@"
