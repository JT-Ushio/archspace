#!/usr/bin/env bash
set -eo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
export MASTER_PORT=${MASTER_PORT:-29812}
exec "$SCRIPT_DIR/_run_common.sh" p2 olmo3-1b-ug-partial-p2-s480-p80-linear-down-rms-fp8-30b "$@"
