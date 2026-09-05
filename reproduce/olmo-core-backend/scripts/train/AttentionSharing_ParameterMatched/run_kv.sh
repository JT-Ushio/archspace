#!/usr/bin/env bash
set -eo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export MASTER_PORT="${MASTER_PORT:-29655}"
exec "${SCRIPT_DIR}/_run_common.sh" kv "olmo3-1b-attn-parammatched-kv-k680-v680" "$@"
