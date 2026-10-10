#!/usr/bin/env bash
set -eo pipefail
if [[ $# -lt 2 ]]; then
  echo "usage: $0 CONFIG_NAME DEFAULT_RUN_NAME [overrides...]" >&2
  exit 2
fi
CONFIG_NAME="$1"
DEFAULT_RUN_NAME="$2"
shift 2
SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
BACKEND_DIR="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
source "$BACKEND_DIR/scripts/prepare/prepare_olmo_gpfs2.sh"
HCEPH_START=/mnt/shared-storage-user/jitao/.local/bin/hceph_start
if [[ ! -f "$HCEPH_START" ]]; then
  echo "Missing hceph mount helper: $HCEPH_START" >&2
  exit 1
fi
source "$HCEPH_START"
set -u

# Use node-local compiler caches rather than the previously exhausted shared tmp.
TASK_CACHE_ROOT="$(mktemp -d /tmp/mose-dense-ffn.XXXXXXXX)"
mkdir -p "$TASK_CACHE_ROOT/tmp" "$TASK_CACHE_ROOT/inductor" "$TASK_CACHE_ROOT/triton" "$TASK_CACHE_ROOT/cuda"
export TMPDIR="$TASK_CACHE_ROOT/tmp" TEMP="$TASK_CACHE_ROOT/tmp" TMP="$TASK_CACHE_ROOT/tmp"
export TORCHINDUCTOR_CACHE_DIR="$TASK_CACHE_ROOT/inductor"
export TRITON_CACHE_DIR="$TASK_CACHE_ROOT/triton" CUDA_CACHE_PATH="$TASK_CACHE_ROOT/cuda"

cd "$MOSE_JT"
RUN_NAME="${RUN_NAME:-$DEFAULT_RUN_NAME}"
STAGE="${STAGE:-stage1}"
PROC_PER_NODE="${PROC_PER_NODE:-8}"
NODE_COUNT="${NODE_COUNT:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29830}"
export WANDB_API_KEY="$WANDB_SJY"

exec "$OLMO_ENV/bin/python" -m torch.distributed.run \
  --nproc_per_node="$PROC_PER_NODE" --nnodes="$NODE_COUNT" \
  --node_rank="$NODE_RANK" --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
  "cfps/BS256_Seq8K_30B_WSD_FP8/Dense_FFN_Linear_RMS/$CONFIG_NAME.py" \
  --name "$RUN_NAME" --sequence-length=8192 --data-root="$DATA_DIR" \
  --save-folder="$CKPT_DIR/$RUN_NAME/$STAGE" --work-dir "$WORK_DIR" \
  --data_loader.global_batch_size=2097152 --train_module.rank_microbatch_size=32768 \
  --train_module.optim.lr=0.005 --train_module.scheduler.warmup=2000 \
  --train_module.scheduler.decay_fraction=0.2 --train_module.float8_config.enabled=true \
  --trainer.max_duration.value=30000000000 --trainer.max_duration.unit=tokens \
  --trainer.callbacks.checkpointer.save_interval=1788 \
  --trainer.callbacks.checkpointer.max_checkpoints=null \
  --init_seed=12536 --data_loader.seed=34521 trainer.work_dir="$WORK_DIR" "$@"
