#!/usr/bin/env bash
set -eo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export MASTER_PORT="${MASTER_PORT:-29654}"
exec "${SCRIPT_DIR}/_run_common.sh" qkv "olmo3-1b-attn-parammatched-qkv-r768" "$@"
