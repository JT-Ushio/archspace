#!/usr/bin/env bash
set -eo pipefail
MOSE_REFERENCE_SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
MOSE_REFERENCE_BACKEND_DIR="$(cd -- "$MOSE_REFERENCE_SCRIPT_DIR/../../.." && pwd)"
export PYTHONDONTWRITEBYTECODE=1
MOSE_REFERENCE_MANIFEST_PATH=${MOSE_REFERENCE_MANIFEST_PATH:-$MOSE_REFERENCE_SCRIPT_DIR/completed_reference_candidates.json}
source "$MOSE_REFERENCE_BACKEND_DIR/scripts/prepare/prepare_olmo_gpfs2.sh"
source /mnt/shared-storage-user/jitao/.local/bin/hceph_start
set -u
MOSE_REFERENCE_RUNTIME_DIR=$(mktemp -d /tmp/mose-reference-export.XXXXXXXX)
export TMPDIR="$MOSE_REFERENCE_RUNTIME_DIR"
echo "Read-only completed-candidate/archived-reference comparison; single CPU process."
exec "$OLMO_ENV/bin/python" "$MOSE_REFERENCE_SCRIPT_DIR/export_with_references.py" \
  --manifest "$MOSE_REFERENCE_MANIFEST_PATH" \
  --output "$MOSE_REFERENCE_RUNTIME_DIR/results.json"
