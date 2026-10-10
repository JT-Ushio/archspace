#!/usr/bin/env bash
set -eo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
export MASTER_PORT=${MASTER_PORT:-29733}
exec "$SCRIPT_DIR/_run_common.sh" s3 olmo3-1b-dense-sharing-s3-linear-fp8-30b "$@"
