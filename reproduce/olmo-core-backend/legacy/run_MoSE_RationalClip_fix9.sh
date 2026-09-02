#!/bin/bash

source /mnt/shared-storage-user/jitao/.local/bin/hceph_start
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/shared-storage-user/jitao/envs/MoSE
# CUDA 12.8: set in $CONDA_PREFIX/etc/conda/activate.d/cuda.sh
# Environment Variables: set in $CONDA_PREFIX/etc/conda/activate.d/prepare.sh

cd /mnt/shared-storage-user/jitao/code/archspace-MoSE/reproduce/olmo-core-backend
RUN_NAME=olmo3-1b-MoSE-nolinear-rmsdown-weight-fp8
STAGE=stage1
WANDB_API_KEY=$WANDB_JT \
torchrun \
  --nproc_per_node=$PROC_PER_NODE \
  --nnodes=$NODE_COUNT \
  --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR \
  --master_port=${MASTER_PORT:-29634} \
  cfgs/OLMo3-1B-stage1-mose-rmsnorm.py \
  --name ${RUN_NAME} \
  --data-root=${DATA_DIR} \
  --save-folder=${CKPT_DIR}/${RUN_NAME}/${STAGE} \
  --work-dir ${WORK_DIR} \
  --model.block.feed_forward.r1=0 \
  --model.block.feed_forward.r2=512 \
  --model.block.feed_forward.down_r1=0 \
  --model.block.feed_forward.down_r2=512 \
  --model.block.feed_forward.gate_nonlinearity=silu \
  --model.block.feed_forward.up_nonlinearity=silu \
  --model.block.feed_forward.down_nonlinearity=rms_norm \
  --model.block.feed_forward.rms_norm_learnable_weight=true \
  --train_module.float8_config.enabled=true \
  trainer.work_dir=${WORK_DIR}
