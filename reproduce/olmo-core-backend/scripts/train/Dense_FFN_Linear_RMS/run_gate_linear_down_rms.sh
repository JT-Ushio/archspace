#!/usr/bin/env bash
set -eo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
export MASTER_PORT="${MASTER_PORT:-29832}"
exec "$SCRIPT_DIR/_run_common.sh" gate_linear_down_rms \
  olmo3-1b-dense-nearmatched-gate-linear-r1632-down-rms-r1632-fp8-30b "$@"
