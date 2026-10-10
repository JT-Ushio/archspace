#!/usr/bin/env bash
# Pool scheduling candidate; isolate output paths from the non-pool candidate.
set -eo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
export RUN_NAME=olmo3-1b-r512matched-c8-linear-down-rms-gateup-fp8-30b-pool
exec "$SCRIPT_DIR/run_c8_linear_down_rms_gateup.sh" "$@"
