#!/usr/bin/env bash
set -eo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
export MASTER_PORT=${MASTER_PORT:-29822}
exec "$SCRIPT_DIR/_run_common.sh" f3 olmo3-1b-ffn-budget-f3-ug480-down672-shared-linear-rms-fp8-30b "$@"
