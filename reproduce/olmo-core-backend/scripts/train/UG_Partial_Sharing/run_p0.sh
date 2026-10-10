#!/usr/bin/env bash
set -eo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
export MASTER_PORT=${MASTER_PORT:-29810}
exec "$SCRIPT_DIR/_run_common.sh" p0 olmo3-1b-ug-partial-p0-s0-p512-linear-down-rms-fp8-30b "$@"
