#!/usr/bin/env bash
set -eo pipefail
MOSE_WIDTH_ANALYSIS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MOSE_WIDTH_BACKEND="$(cd -- "$MOSE_WIDTH_ANALYSIS_DIR/../../.." && pwd)"
source "$MOSE_WIDTH_BACKEND/scripts/prepare/prepare_olmo_gpfs2.sh"
source /mnt/shared-storage-user/jitao/.local/bin/hceph_start
set -u
[[ "${NODE_COUNT:-1}" == 1 && "${PROC_PER_NODE:-8}" == 8 ]] || exit 2
MOSE_WIDTH_OUTPUT="$MOSE_WIDTH_ANALYSIS_DIR/analysis/completed_20261009"
MOSE_WIDTH_RUNTIME="$(mktemp -d /tmp/mose-width-analysis.XXXXXXXX)"
export TMPDIR="$MOSE_WIDTH_RUNTIME/tmp"
export TORCHINDUCTOR_CACHE_DIR="$MOSE_WIDTH_RUNTIME/inductor"
export TRITON_CACHE_DIR="$MOSE_WIDTH_RUNTIME/triton"
export CUDA_CACHE_PATH="$MOSE_WIDTH_RUNTIME/cuda"
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$TMPDIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH"
"$OLMO_ENV/bin/python" "$MOSE_WIDTH_ANALYSIS_DIR/analyze_histories.py" --output-dir "$MOSE_WIDTH_OUTPUT"
MOSE_WIDTH_PORT=30141
mkdir -p "$MOSE_WIDTH_OUTPUT/validation"
for MOSE_WIDTH_CASE in w1 w1qk w1m w2 g1; do
  "$OLMO_ENV/bin/python" -m torch.distributed.run \
    --nproc_per_node=8 --nnodes=1 --node_rank=0 \
    --master_addr="${MASTER_ADDR:-127.0.0.1}" --master_port="$MOSE_WIDTH_PORT" \
    "$MOSE_WIDTH_BACKEND/scripts/analysis/Corrected_LM_Evaluation/evaluate.py" \
    --case "$MOSE_WIDTH_CASE" --manifest "$MOSE_WIDTH_ANALYSIS_DIR/completed_cases.json" \
    --data-root "$DATA_DIR" --runtime-dir "$MOSE_WIDTH_RUNTIME/$MOSE_WIDTH_CASE"
  cp "$MOSE_WIDTH_RUNTIME/$MOSE_WIDTH_CASE/results.json" "$MOSE_WIDTH_OUTPUT/validation/$MOSE_WIDTH_CASE.json"
  MOSE_WIDTH_PORT=$((MOSE_WIDTH_PORT + 1))
done
echo "MOSE_WIDTH_ANALYSIS_COMPLETE $MOSE_WIDTH_OUTPUT"
