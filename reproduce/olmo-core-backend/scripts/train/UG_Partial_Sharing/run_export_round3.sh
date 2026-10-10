#!/usr/bin/env bash
set -eo pipefail
MOSE_ROUND3_SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
export MOSE_REFERENCE_MANIFEST_PATH="$MOSE_ROUND3_SCRIPT_DIR/completed_reference_candidates_round3.json"
exec "$MOSE_ROUND3_SCRIPT_DIR/run_export_with_references.sh"
