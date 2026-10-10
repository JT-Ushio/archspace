#!/usr/bin/env bash
set -eo pipefail
MOSE_ROUND5_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
MOSE_ROUND5_BACKEND="$(cd -- "$MOSE_ROUND5_DIR/../../.." && pwd)"
MOSE_ROUND5_MANIFEST="$MOSE_ROUND5_DIR/completed_reference_candidates_round5.json"
export PYTHONDONTWRITEBYTECODE=1
source "$MOSE_ROUND5_BACKEND/scripts/prepare/prepare_olmo_gpfs2.sh"
source /mnt/shared-storage-user/jitao/.local/bin/hceph_start
set -u
MOSE_ROUND5_RUNTIME=$(mktemp -d /tmp/mose-completed-round5.XXXXXXXX)
export TMPDIR="$MOSE_ROUND5_RUNTIME"
echo "Read-only completed cohort including F4qk/F5qk, then archived C0/C2/C6 references; CPU export inside scheduled allocation."
"$OLMO_ENV/bin/python" "$MOSE_ROUND5_DIR/export_completed_batch.py" \
  --cohort --manifest "$MOSE_ROUND5_MANIFEST" \
  --output "$MOSE_ROUND5_RUNTIME/completed_cohort.json"
exec "$OLMO_ENV/bin/python" "$MOSE_ROUND5_DIR/export_with_references.py" \
  --manifest "$MOSE_ROUND5_MANIFEST" \
  --output "$MOSE_ROUND5_RUNTIME/with_references.json"
