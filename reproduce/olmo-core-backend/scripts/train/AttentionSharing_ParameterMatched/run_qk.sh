#!/usr/bin/env bash
set -eo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export MASTER_PORT="${MASTER_PORT:-29652}"
exec "${SCRIPT_DIR}/_run_common.sh" qk "olmo3-1b-attn-parammatched-qk-q680" "$@"
