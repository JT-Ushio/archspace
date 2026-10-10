#!/usr/bin/env bash
set -eo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
export MASTER_PORT=${MASTER_PORT:-29813}
exec "$SCRIPT_DIR/_run_common.sh" p3 olmo3-1b-ug-partial-p3-s320-p224-linear-down-rms-fp8-30b "$@"
