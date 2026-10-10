#!/usr/bin/env bash
set -eo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 CONFIG_NAME DEFAULT_RUN_NAME [model safeguards...]" >&2
  exit 2
fi

CONFIG_NAME="$1"
DEFAULT_RUN_NAME="$2"
shift 2
MOSE_CONFIG_OVERRIDES=("$@")

SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
BACKEND_DIR="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
source "$BACKEND_DIR/scripts/prepare/prepare_olmo_gpfs2.sh"
set -u

# Compiler files are disposable and belong on this job's local filesystem.
# The shared jitao/tmp_data quota caused GCC/Inductor failures in attempt r1.
export MOSE_RUNTIME_CACHE_ROOT="${MOSE_RUNTIME_CACHE_ROOT:-$(mktemp -d /tmp/mose-runtime.XXXXXXXX)}"
mkdir -p "$MOSE_RUNTIME_CACHE_ROOT/tmp" "$MOSE_RUNTIME_CACHE_ROOT/inductor" \
  "$MOSE_RUNTIME_CACHE_ROOT/triton" "$MOSE_RUNTIME_CACHE_ROOT/cuda"
export TMPDIR="$MOSE_RUNTIME_CACHE_ROOT/tmp"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export TORCHINDUCTOR_CACHE_DIR="$MOSE_RUNTIME_CACHE_ROOT/inductor"
export TRITON_CACHE_DIR="$MOSE_RUNTIME_CACHE_ROOT/triton"
export CUDA_CACHE_PATH="$MOSE_RUNTIME_CACHE_ROOT/cuda"
echo "Job-local compiler cache: $MOSE_RUNTIME_CACHE_ROOT"
df -h "$MOSE_RUNTIME_CACHE_ROOT"


HCEPH_START=/mnt/shared-storage-user/jitao/.local/bin/hceph_start
if [[ ! -f "$HCEPH_START" ]]; then
  echo "Missing hceph mount helper: $HCEPH_START" >&2
  exit 1
fi
source "$HCEPH_START"

if [[ -n "${MOSE_CHECKPOINT_ROOT:-}" ]]; then
  [[ -d /mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji ]] || exit 1
  export CKPT_DIR="$MOSE_CHECKPOINT_ROOT"
  mkdir -p "$CKPT_DIR"
  echo "Checkpoint storage override: $CKPT_DIR"
  df -h "$CKPT_DIR"
  MOSE_CHECKPOINT_FREE_BYTES=$(df -B1 --output=avail "$CKPT_DIR" | tail -n 1 | tr -d '[:space:]')
  if [[ ! "$MOSE_CHECKPOINT_FREE_BYTES" =~ ^[0-9]+$ ]] || (( MOSE_CHECKPOINT_FREE_BYTES < 68719476736 )); then
    echo "Checkpoint filesystem has less than 64 GiB available; not starting training"
    exit 1
  fi
  MOSE_STORAGE_PROBE=$(mktemp "$CKPT_DIR/.mose-storage-probe.XXXXXXXX")
  if ! dd if=/dev/zero of="$MOSE_STORAGE_PROBE" bs=1048576 count=16 conv=fsync status=none; then
    rm -f -- "$MOSE_STORAGE_PROBE"
    exit 1
  fi
  rm -f -- "$MOSE_STORAGE_PROBE"
fi

cd "$MOSE_JT"

if [[ ! -v RUN_NAME ]]; then RUN_NAME="$DEFAULT_RUN_NAME"; fi
if [[ ! -v STAGE ]]; then STAGE=stage1; fi
if [[ -n "${MOSE_STAGE_OVERRIDE:-}" ]]; then STAGE="$MOSE_STAGE_OVERRIDE"; fi
if [[ ! -v PROC_PER_NODE ]]; then PROC_PER_NODE=8; fi
if [[ ! -v NODE_COUNT ]]; then NODE_COUNT=1; fi
if [[ ! -v NODE_RANK ]]; then NODE_RANK=0; fi
if [[ ! -v MASTER_ADDR ]]; then MASTER_ADDR=127.0.0.1; fi
if [[ ! -v MASTER_PORT ]]; then MASTER_PORT=29700; fi

export WANDB_API_KEY="$WANDB_SJY"

exec "$OLMO_ENV/bin/python" -m torch.distributed.run \
  --nproc_per_node="$PROC_PER_NODE" \
  --nnodes="$NODE_COUNT" \
  --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  "cfps/BS256_Seq8K_30B_WSD_FP8/UG_Partial_Sharing/$CONFIG_NAME.py" \
  --name "$RUN_NAME" \
  --sequence-length=8192 \
  --data-root="$DATA_DIR" \
  --save-folder="$CKPT_DIR/$RUN_NAME/$STAGE" \
  --work-dir "$WORK_DIR" \
  --data_loader.global_batch_size=2097152 \
  --train_module.rank_microbatch_size=32768 \
  --train_module.optim.lr=0.005 \
  --train_module.scheduler.warmup=2000 \
  --train_module.scheduler.decay_fraction=0.2 \
  --train_module.float8_config.enabled=true \
  --trainer.max_duration.value=30000000000 \
  --trainer.max_duration.unit=tokens \
  --trainer.callbacks.checkpointer.save_interval=1788 \
  --trainer.callbacks.checkpointer.max_checkpoints=null \
  --init_seed=12536 \
  --data_loader.seed=34521 \
  trainer.work_dir="$WORK_DIR" \
  "${MOSE_CONFIG_OVERRIDES[@]}"
