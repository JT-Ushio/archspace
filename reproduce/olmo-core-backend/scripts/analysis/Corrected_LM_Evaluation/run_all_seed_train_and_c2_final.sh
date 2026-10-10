#!/usr/bin/env bash
set -euo pipefail
MOSE_SEED_COMBINED_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
[[ "${NODE_COUNT:-1}" == 1 && "${PROC_PER_NODE:-8}" == 8 ]] || exit 2

# Run both read-only analyses in one allocation. The first stage requires
# verified final completion for all six training histories; the second checks
# the actual final checkpoint metadata before evaluating each C2 replica.
bash "$MOSE_SEED_COMBINED_DIR/../../train/UG_Partial_Sharing/run_export_seed_confirmation.sh"
exec bash "$MOSE_SEED_COMBINED_DIR/run_c2_seed_final.sh"
