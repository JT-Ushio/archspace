#!/usr/bin/env bash
set -eo pipefail
MOSE_LM_SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
MOSE_LM_BACKEND_DIR="$(cd -- "$MOSE_LM_SCRIPT_DIR/../../.." && pwd)"
export PYTHONDONTWRITEBYTECODE=1
source "$MOSE_LM_BACKEND_DIR/scripts/prepare/prepare_olmo_gpfs2.sh"
source /mnt/shared-storage-user/jitao/.local/bin/hceph_start
set -u
MOSE_LM_RUNTIME_ROOT=$(mktemp -d /tmp/mose-corrected-lm.XXXXXXXX)
export TMPDIR="$MOSE_LM_RUNTIME_ROOT/tmp"
export TORCHINDUCTOR_CACHE_DIR="$MOSE_LM_RUNTIME_ROOT/inductor"
export TRITON_CACHE_DIR="$MOSE_LM_RUNTIME_ROOT/triton"
export CUDA_CACHE_PATH="$MOSE_LM_RUNTIME_ROOT/cuda"
mkdir -p "$TMPDIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH"
[[ "${NODE_COUNT:-1}" == 1 && "${PROC_PER_NODE:-8}" == 8 ]] || exit 2
MOSE_LM_PORT=30071
for MOSE_LM_CASE in c0s2 c6s2 c0s3 c6s3; do
  "$OLMO_ENV/bin/python" -m torch.distributed.run \
    --nproc_per_node=8 --nnodes=1 --node_rank=0 \
    --master_addr="${MASTER_ADDR:-127.0.0.1}" --master_port="$MOSE_LM_PORT" \
    "$MOSE_LM_SCRIPT_DIR/evaluate.py" --case "$MOSE_LM_CASE" --manifest "$MOSE_LM_SCRIPT_DIR/c0_c6_seed_cases.json" \
    --data-root "$DATA_DIR" --runtime-dir "$MOSE_LM_RUNTIME_ROOT/$MOSE_LM_CASE"
  MOSE_LM_PORT=$((MOSE_LM_PORT + 1))
done
