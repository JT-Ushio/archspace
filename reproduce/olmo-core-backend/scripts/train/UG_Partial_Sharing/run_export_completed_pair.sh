#!/usr/bin/env bash
set -eo pipefail
MOSE_PAIR_SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
export MOSE_EXPORT_COHORT=true
export MOSE_EXPORT_MANIFEST_PATH="$MOSE_PAIR_SCRIPT_DIR/completed_pair_segments.json"
exec "$MOSE_PAIR_SCRIPT_DIR/run_export_completed_batch.sh"
