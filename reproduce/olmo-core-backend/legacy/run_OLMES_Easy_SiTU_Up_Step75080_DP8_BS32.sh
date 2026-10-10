#!/usr/bin/env bash
set -euo pipefail

# Compatibility entry for the old checkpoint; use the generic optimized launcher.
BACKEND_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
GPFS_ROOT="${OLMES_GPFS_ROOT:-/mnt/shared-storage-gpfs2/intern-pretrain-shared02/shijiayang-p}"
if [[ ! -d "${GPFS_ROOT}/conda_envs/olmes" ]]; then
    GPFS_ROOT="/mnt/shared-storage-user/intern-pretrain-shared02/shijiayang-p"
fi
export OLMES_GPFS_ROOT="${GPFS_ROOT}"
export OLMES_MODEL_DIR="${OLMES_MODEL_DIR:-${GPFS_ROOT}/ckpts/huggingface/olmo3-1b-situ-up-step75080}"
exec bash "${BACKEND_ROOT}/scripts/eval/OLMES_Easy/run_easy.sh" "$@"
