#!/usr/bin/env bash
set -eo pipefail
MOSE_EFF_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
MOSE_EFF_BACKEND="$(cd -- "$MOSE_EFF_DIR/../../.." && pwd)"
export PYTHONDONTWRITEBYTECODE=1
source "$MOSE_EFF_BACKEND/scripts/prepare/prepare_olmo_gpfs2.sh"
source /mnt/shared-storage-user/jitao/.local/bin/hceph_start
set -u
MOSE_EFF_RUNTIME=$(mktemp -d /tmp/mose-training-efficiency.XXXXXXXX)
[[ "${NODE_COUNT:-1}" == 1 && "${PROC_PER_NODE:-8}" == 8 ]] || exit 2
MOSE_EFF_PORT=30061
for MOSE_EFF_CASE in dense c0 c6; do
  export TMPDIR="$MOSE_EFF_RUNTIME/$MOSE_EFF_CASE/tmp"
  export TORCHINDUCTOR_CACHE_DIR="$MOSE_EFF_RUNTIME/$MOSE_EFF_CASE/inductor"
  export TRITON_CACHE_DIR="$MOSE_EFF_RUNTIME/$MOSE_EFF_CASE/triton"
  export CUDA_CACHE_PATH="$MOSE_EFF_RUNTIME/$MOSE_EFF_CASE/cuda"
  mkdir -p "$TMPDIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH"
  "$OLMO_ENV/bin/python" -m torch.distributed.run \
    --nproc_per_node=8 --nnodes=1 --node_rank=0 \
    --master_addr="${MASTER_ADDR:-127.0.0.1}" --master_port="$MOSE_EFF_PORT" \
    "$MOSE_EFF_DIR/benchmark.py" --case "$MOSE_EFF_CASE" \
    --data-root "$DATA_DIR" --runtime-dir "$MOSE_EFF_RUNTIME/$MOSE_EFF_CASE"
  MOSE_EFF_PORT=$((MOSE_EFF_PORT + 1))
done
