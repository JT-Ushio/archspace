#!/usr/bin/env bash
set -eo pipefail
MOSE_ROUND8_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
MOSE_ROUND8_BACKEND="$(cd -- "$MOSE_ROUND8_DIR/../../.." && pwd)"
MOSE_ROUND8_MANIFEST="$MOSE_ROUND8_DIR/completed_reference_candidates_round8.json"
export PYTHONDONTWRITEBYTECODE=1
source "$MOSE_ROUND8_BACKEND/scripts/prepare/prepare_olmo_gpfs2.sh"
source /mnt/shared-storage-user/jitao/.local/bin/hceph_start
set -u
MOSE_ROUND8_RUNTIME=$(mktemp -d /tmp/mose-completed-round8.XXXXXXXX)
export TMPDIR="$MOSE_ROUND8_RUNTIME"
echo "Read-only completed cohort including completed A1/A2, then archived C0/C2/C6 references; CPU export inside scheduled allocation."
"$OLMO_ENV/bin/python" "$MOSE_ROUND8_DIR/export_completed_batch.py" \
  --cohort --manifest "$MOSE_ROUND8_MANIFEST" \
  --output "$MOSE_ROUND8_RUNTIME/completed_cohort.json"
exec "$OLMO_ENV/bin/python" "$MOSE_ROUND8_DIR/export_with_references.py" \
  --manifest "$MOSE_ROUND8_MANIFEST" \
  --output "$MOSE_ROUND8_RUNTIME/with_references.json"
