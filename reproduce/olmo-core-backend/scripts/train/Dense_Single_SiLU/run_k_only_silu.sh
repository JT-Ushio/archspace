#!/usr/bin/env bash
set -eo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
export MASTER_PORT=${MASTER_PORT:-29722}
exec "$SCRIPT_DIR/_run_common.sh" k_only_silu olmo3-1b-dense-matched-k-only-silu-r1024-fp8-30b "$@"
