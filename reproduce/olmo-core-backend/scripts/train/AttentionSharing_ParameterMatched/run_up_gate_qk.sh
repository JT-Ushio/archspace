#!/usr/bin/env bash
set -eo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export MASTER_PORT="${MASTER_PORT:-29653}"
exec "${SCRIPT_DIR}/_run_common.sh" up_gate_qk "olmo3-1b-attn-parammatched-up-gate-qk-r568-q680" "$@"
