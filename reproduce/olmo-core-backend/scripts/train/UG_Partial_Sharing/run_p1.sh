#!/usr/bin/env bash
set -eo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
export MASTER_PORT=${MASTER_PORT:-29811}
exec "$SCRIPT_DIR/_run_common.sh" p1 olmo3-1b-ug-partial-p1-s560-p0-linear-down-rms-fp8-30b "$@"
