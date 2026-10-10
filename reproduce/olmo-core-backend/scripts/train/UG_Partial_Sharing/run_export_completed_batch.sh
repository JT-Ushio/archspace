#!/usr/bin/env bash
set -eo pipefail
MOSE_EXPORT_SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
MOSE_EXPORT_BACKEND_DIR="$(cd -- "$MOSE_EXPORT_SCRIPT_DIR/../../.." && pwd)"
export PYTHONDONTWRITEBYTECODE=1
MOSE_EXPORT_MANIFEST_PATH=${MOSE_EXPORT_MANIFEST_PATH:-$MOSE_EXPORT_SCRIPT_DIR/batch_segments.json}
MOSE_EXPORT_SCOPE_FLAGS=()
if [[ "${MOSE_EXPORT_COHORT:-false}" == true ]]; then MOSE_EXPORT_SCOPE_FLAGS=(--cohort); fi
source "$MOSE_EXPORT_BACKEND_DIR/scripts/prepare/prepare_olmo_gpfs2.sh"
source /mnt/shared-storage-user/jitao/.local/bin/hceph_start
set -u
MOSE_EXPORT_RUNTIME_DIR=$(mktemp -d /tmp/mose-batch-export.XXXXXXXX)
export TMPDIR="$MOSE_EXPORT_RUNTIME_DIR"
echo "Read-only W&B history comparison; single CPU process in a scheduled 8-GPU allocation."
exec "$OLMO_ENV/bin/python" "$MOSE_EXPORT_SCRIPT_DIR/export_completed_batch.py" \
  --manifest "$MOSE_EXPORT_MANIFEST_PATH" \
  "${MOSE_EXPORT_SCOPE_FLAGS[@]}" \
  --output "$MOSE_EXPORT_RUNTIME_DIR/results.json"
