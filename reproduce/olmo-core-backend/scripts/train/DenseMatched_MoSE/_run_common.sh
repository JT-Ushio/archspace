#!/usr/bin/env bash
set -eo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 CONFIG_NAME DEFAULT_RUN_NAME [model safeguards...]" >&2
  exit 2
fi

CONFIG_NAME="$1"
DEFAULT_RUN_NAME="$2"
shift 2

SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
BACKEND_DIR="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
source "$BACKEND_DIR/scripts/prepare/prepare_olmo_gpfs2.sh"
set -u

HCEPH_START=/mnt/shared-storage-user/jitao/.local/bin/hceph_start
if [[ ! -f "$HCEPH_START" ]]; then
  echo "Missing hceph mount helper: $HCEPH_START" >&2
  exit 1
fi
source "$HCEPH_START"

cd "$MOSE_JT"

if [[ ! -v RUN_NAME ]]; then RUN_NAME="$DEFAULT_RUN_NAME"; fi
if [[ ! -v STAGE ]]; then STAGE=stage1; fi
if [[ ! -v PROC_PER_NODE ]]; then PROC_PER_NODE=8; fi
if [[ ! -v NODE_COUNT ]]; then NODE_COUNT=1; fi
if [[ ! -v NODE_RANK ]]; then NODE_RANK=0; fi
if [[ ! -v MASTER_ADDR ]]; then MASTER_ADDR=127.0.0.1; fi
if [[ ! -v MASTER_PORT ]]; then MASTER_PORT=29680; fi

export WANDB_API_KEY="$WANDB_SJY"

exec "$OLMO_ENV/bin/python" -m torch.distributed.run \
  --nproc_per_node="$PROC_PER_NODE" \
  --nnodes="$NODE_COUNT" \
  --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  "cfps/BS256_Seq8K_30B_WSD_FP8/DenseMatched_MoSE/$CONFIG_NAME.py" \
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
  trainer.work_dir="$WORK_DIR" \
  "$@"
