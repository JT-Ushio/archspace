#!/usr/bin/env bash
set -eo pipefail
MOSE_SEED_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
MOSE_SEED_BACKEND="$(cd -- "$MOSE_SEED_DIR/../../.." && pwd)"
MOSE_SEED_MANIFEST="$MOSE_SEED_DIR/completed_seed_segments.json"
export PYTHONDONTWRITEBYTECODE=1
source "$MOSE_SEED_BACKEND/scripts/prepare/prepare_olmo_gpfs2.sh"
source /mnt/shared-storage-user/jitao/.local/bin/hceph_start
set -u
MOSE_SEED_RUNTIME=$(mktemp -d /tmp/mose-seed-confirmation.XXXXXXXX)
export TMPDIR="$MOSE_SEED_RUNTIME"
echo "Read-only completed C0/C2/C6 seed12537/12538 histories; paired initialization-seed confirmation with fixed data order."
exec "$OLMO_ENV/bin/python" "$MOSE_SEED_DIR/export_completed_batch.py" \
  --cohort --manifest "$MOSE_SEED_MANIFEST" \
  --output "$MOSE_SEED_RUNTIME/completed_cohort.json"
