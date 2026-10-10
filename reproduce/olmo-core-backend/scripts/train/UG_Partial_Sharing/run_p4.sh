#!/usr/bin/env bash
set -eo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
export MASTER_PORT=${MASTER_PORT:-29814}
exec "$SCRIPT_DIR/_run_common.sh" p4 olmo3-1b-ug-partial-p4-s160-p368-linear-down-rms-fp8-30b "$@"
