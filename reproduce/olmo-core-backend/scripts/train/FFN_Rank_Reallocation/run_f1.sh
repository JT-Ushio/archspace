#!/usr/bin/env bash
set -eo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
export MASTER_PORT=${MASTER_PORT:-29820}
exec "$SCRIPT_DIR/_run_common.sh" f1 olmo3-1b-ffn-budget-f1-ug384-down768-unshared-linear-rms-fp8-30b "$@"
