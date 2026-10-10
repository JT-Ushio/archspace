#!/usr/bin/env bash
set -eo pipefail
MOSE_REF_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
MOSE_REF_BACKEND="$(cd -- "$MOSE_REF_DIR/../../.." && pwd)"
export PYTHONDONTWRITEBYTECODE=1
source "$MOSE_REF_BACKEND/scripts/prepare/prepare_olmo_gpfs2.sh"
source /mnt/shared-storage-user/jitao/.local/bin/hceph_start
set -u
MOSE_REF_RUNTIME=$(mktemp -d /tmp/mose-reference-lm.XXXXXXXX)
export TMPDIR="$MOSE_REF_RUNTIME/tmp"
export TORCHINDUCTOR_CACHE_DIR="$MOSE_REF_RUNTIME/inductor"
export TRITON_CACHE_DIR="$MOSE_REF_RUNTIME/triton"
export CUDA_CACHE_PATH="$MOSE_REF_RUNTIME/cuda"
mkdir -p "$TMPDIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH"
[[ "${NODE_COUNT:-1}" == 1 && "${PROC_PER_NODE:-8}" == 8 ]] || exit 2
MOSE_REF_PORT=30051
for MOSE_REF_CASE in c0 c2 c6; do
  "$OLMO_ENV/bin/python" -m torch.distributed.run \
    --nproc_per_node=8 --nnodes=1 --node_rank=0 \
    --master_addr="${MASTER_ADDR:-127.0.0.1}" --master_port="$MOSE_REF_PORT" \
    "$MOSE_REF_DIR/evaluate.py" --case "$MOSE_REF_CASE" \
    --manifest "$MOSE_REF_DIR/reference_cases.json" \
    --data-root "$DATA_DIR" --runtime-dir "$MOSE_REF_RUNTIME/$MOSE_REF_CASE"
  MOSE_REF_PORT=$((MOSE_REF_PORT + 1))
done
