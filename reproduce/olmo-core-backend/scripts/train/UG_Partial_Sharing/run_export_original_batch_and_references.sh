#!/usr/bin/env bash
set -eo pipefail
MOSE_FINAL_SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
MOSE_FINAL_BACKEND_DIR="$(cd -- "$MOSE_FINAL_SCRIPT_DIR/../../.." && pwd)"
MOSE_FINAL_BATCH_MANIFEST="$MOSE_FINAL_SCRIPT_DIR/batch_segments.json"
MOSE_FINAL_REFERENCE_MANIFEST="$MOSE_FINAL_SCRIPT_DIR/completed_reference_candidates_round4.json"
export PYTHONDONTWRITEBYTECODE=1
source "$MOSE_FINAL_BACKEND_DIR/scripts/prepare/prepare_olmo_gpfs2.sh"
source /mnt/shared-storage-user/jitao/.local/bin/hceph_start
set -u
MOSE_FINAL_RUNTIME_DIR=$(mktemp -d /tmp/mose-final-batch-export.XXXXXXXX)
export TMPDIR="$MOSE_FINAL_RUNTIME_DIR"
echo "Read-only original ten-config batch and completed candidates with references; single CPU process."
"$OLMO_ENV/bin/python" "$MOSE_FINAL_SCRIPT_DIR/export_completed_batch.py" \
  --manifest "$MOSE_FINAL_BATCH_MANIFEST" \
  --output "$MOSE_FINAL_RUNTIME_DIR/batch.json"
exec "$OLMO_ENV/bin/python" "$MOSE_FINAL_SCRIPT_DIR/export_with_references.py" \
  --manifest "$MOSE_FINAL_REFERENCE_MANIFEST" \
  --output "$MOSE_FINAL_RUNTIME_DIR/with_references.json"
