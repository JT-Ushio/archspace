#!/usr/bin/env bash
set -euo pipefail

# Independent rerun of the down-RMSNorm ablation.  Keep the architecture and
# all training overrides in the canonical script; only use a separate default
# run/checkpoint name so this rerun cannot collide with the original output.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export RUN_NAME="${RUN_NAME:-mose-down-rms-rerun}"

exec "${SCRIPT_DIR}/run_OLMo3_1B_FullLowRank_QKVO_r512_IndependentGateUp_r512_Down_r512_RMSNorm_BS256_Seq8K_30B_WSD_FP8.sh" "$@"
