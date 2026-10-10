#!/usr/bin/env bash
set -eo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
export MASTER_PORT=${MASTER_PORT:-29724}
exec "$SCRIPT_DIR/_run_common.sh" gate_only_silu olmo3-1b-dense-nearmatched-gate-only-silu-r1632-fp8-30b "$@"
