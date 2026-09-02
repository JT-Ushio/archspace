#!/bin/bash

source /mnt/shared-storage-user/jitao/.local/bin/hceph_start
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/shared-storage-user/jitao/envs/MoSE
# CUDA 12.8: set in $CONDA_PREFIX/etc/conda/activate.d/cuda.sh
# Environment Variables: set in $CONDA_PREFIX/etc/conda/activate.d/prepare.sh

cd /mnt/shared-storage-user/jitao/code/archspace-MoSE/reproduce/olmo-core-backend
RUN_NAME=olmo3-1b-RationalClip

WANDB_API_KEY=$WANDB_JT \
torchrun \
  --nproc_per_node=$PROC_PER_NODE \
  --nnodes=$NODE_COUNT \
  --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR \
  --master_port=${MASTER_PORT:-29634} \
  cfgs/OLMo3-1B-stage1-asymmetric-rational-clip.py \
  --name ${RUN_NAME} \
  --data-root=${DATA_DIR} \
  --save-folder=${CKPT_DIR}/${RUN_NAME} \
  --work-dir ${WORK_DIR} \
  trainer.work_dir=${WORK_DIR}
