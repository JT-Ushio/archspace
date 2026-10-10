#!/usr/bin/env bash
set -euo pipefail
export OLMES_OUTPUT_DIR="/mnt/shared-storage-gpfs2/intern-pretrain-shared02/shijiayang-p/olmo3_outputs/evals/olmes/dense-single-silu-step14306-20261008-gpu-20261008-2124-r2/gate"
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/_run_single_silu_eval.sh" gate "$@"
