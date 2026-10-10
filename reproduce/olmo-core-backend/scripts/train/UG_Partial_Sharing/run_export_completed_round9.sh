#!/usr/bin/env bash
set -eo pipefail
MOSE_ROUND9_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
MOSE_ROUND9_BACKEND="$(cd -- "$MOSE_ROUND9_DIR/../../.." && pwd)"
MOSE_ROUND9_MANIFEST="$MOSE_ROUND9_DIR/completed_reference_candidates_round9.json"
export PYTHONDONTWRITEBYTECODE=1
source "$MOSE_ROUND9_BACKEND/scripts/prepare/prepare_olmo_gpfs2.sh"
source /mnt/shared-storage-user/jitao/.local/bin/hceph_start
set -u
MOSE_ROUND9_RUNTIME=$(mktemp -d /tmp/mose-completed-round9.XXXXXXXX)
export TMPDIR="$MOSE_ROUND9_RUNTIME"
echo "Read-only completed cohort including completed F4/F5 and all nonseed candidates, then archived C0/C2/C6 references; CPU export inside scheduled allocation."
"$OLMO_ENV/bin/python" "$MOSE_ROUND9_DIR/export_completed_batch.py" \
  --cohort --manifest "$MOSE_ROUND9_MANIFEST" \
  --output "$MOSE_ROUND9_RUNTIME/completed_cohort.json"
exec "$OLMO_ENV/bin/python" "$MOSE_ROUND9_DIR/export_with_references.py" \
  --manifest "$MOSE_ROUND9_MANIFEST" \
  --output "$MOSE_ROUND9_RUNTIME/with_references.json"
