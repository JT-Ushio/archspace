#!/usr/bin/env bash
set -eo pipefail
MOSE_ATTN_REPORT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
MOSE_ATTN_REPORT_BACKEND="$(cd -- "$MOSE_ATTN_REPORT_DIR/../../.." && pwd)"
export PYTHONDONTWRITEBYTECODE=1
source "$MOSE_ATTN_REPORT_BACKEND/scripts/prepare/prepare_olmo_gpfs2.sh"
source /mnt/shared-storage-user/jitao/.local/bin/hceph_start
set -u
MOSE_ATTN_REPORT_RUNTIME=$(mktemp -d /tmp/mose-attn-report.XXXXXXXX)
export TMPDIR="$MOSE_ATTN_REPORT_RUNTIME"
echo "Read-only complete A1/A2 Train CE export with archived C0/C2/C6; single CPU process inside scheduled allocation."
exec "$OLMO_ENV/bin/python" "$MOSE_ATTN_REPORT_BACKEND/scripts/train/UG_Partial_Sharing/export_with_references.py" \
  --manifest "$MOSE_ATTN_REPORT_DIR/completed_a1_a2_segments.json" \
  --output "$MOSE_ATTN_REPORT_RUNTIME/a1_a2_with_references.json"
