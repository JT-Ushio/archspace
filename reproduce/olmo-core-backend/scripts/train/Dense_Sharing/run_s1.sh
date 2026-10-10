#!/usr/bin/env bash
set -eo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
export MASTER_PORT=${MASTER_PORT:-29731}
exec "$SCRIPT_DIR/_run_common.sh" s1 olmo3-1b-dense-sharing-s1-linear-fp8-30b "$@"
