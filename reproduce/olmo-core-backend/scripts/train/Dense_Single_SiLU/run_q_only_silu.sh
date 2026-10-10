#!/usr/bin/env bash
set -eo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
export MASTER_PORT=${MASTER_PORT:-29721}
exec "$SCRIPT_DIR/_run_common.sh" q_only_silu olmo3-1b-dense-matched-q-only-silu-r1024-fp8-30b "$@"
