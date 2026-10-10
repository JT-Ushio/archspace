#!/usr/bin/env bash
set -eo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
export MASTER_PORT=${MASTER_PORT:-29821}
exec "$SCRIPT_DIR/_run_common.sh" f2 olmo3-1b-ffn-budget-f2-ug256-down1024-unshared-linear-rms-fp8-30b "$@"
