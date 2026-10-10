#!/usr/bin/env bash
set -eo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
export MASTER_PORT=${MASTER_PORT:-29822}
export PYTHONDONTWRITEBYTECODE=1
exec "$SCRIPT_DIR/_run_common.sh" p2qk olmo3-1b-ug-partial-p2qk-s480-p80-linear-qk-silu-down-rms-fp8-30b "$@"
