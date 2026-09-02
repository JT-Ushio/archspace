#!/bin/bash

source /mnt/shared-storage-user/jitao/.local/bin/hceph_start
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/shared-storage-user/jitao/envs/MoSE
# CUDA 12.8: set in $CONDA_PREFIX/etc/conda/activate.d/cuda.sh
# Environment Variables: set in $CONDA_PREFIX/etc/conda/activate.d/prepare.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1
export PYTHONPATH="$SCRIPT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

RUN_NAME=olmo3-1b-mose-cola-r1-512-r2-512-down-r1-256-down-r2-256-swiglu-5e-3
STAGE=stage1

WANDB_API_KEY=$WANDB_SJY \
torchrun \
  --nproc_per_node="$PROC_PER_NODE" \
  --nnodes="$NODE_COUNT" \
  --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_ADDR" \
  --master_port="${MASTER_PORT:-29634}" \
  cfgs/OLMo3-1B-stage1-mose-situ.py \
  --name "$RUN_NAME" \
  --data-root="$DATA_DIR" \
  --save-folder="$CKPT_DIR/$RUN_NAME/$STAGE" \
  --work-dir "$WORK_DIR" \
  --train_module.float8_config.enabled=true \
  --model.block.feed_forward.r1=512 \
  --model.block.feed_forward.r2=512 \
  --model.block.feed_forward.down_r1=256 \
  --model.block.feed_forward.down_r2=256 \
  --model.block.feed_forward.control=standard \
  --model.block.feed_forward.control_scope=none \
  --model.block.feed_forward.gate_nonlinearity=silu \
  --model.block.feed_forward.up_nonlinearity=silu \
  --model.block.feed_forward.down_nonlinearity=silu \
  trainer.work_dir="$WORK_DIR"
