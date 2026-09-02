#!/bin/bash

source /mnt/shared-storage-user/jitao/.local/bin/hceph_start
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/shared-storage-user/jitao/envs/MoSE
# CUDA 12.8: set in $CONDA_PREFIX/etc/conda/activate.d/cuda.sh
# Environment Variables: set in $CONDA_PREFIX/etc/conda/activate.d/prepare.sh

# cd /mnt/shared-storage-user/jitao/code/archspace-MoSE/reproduce/olmo-core-backend
cd /mnt/shared-storage-gpfs2/intern-pretrain-shared02/shijiayang-p/archspace-MoSE/reproduce/olmo-core-backend
RUN_NAME=olmo3-1b-SiTU-gate-5e-3
STAGE=stage1
WANDB_API_KEY=$WANDB_SJY \
torchrun \
  --nproc_per_node=$PROC_PER_NODE \
  --nnodes=$NODE_COUNT \
  --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR \
  --master_port=${MASTER_PORT:-29634} \
  cfgs/OLMo3-1B-stage1-situ.py \
  --name ${RUN_NAME} \
  --data-root=${DATA_DIR} \
  --save-folder=${CKPT_DIR}/${RUN_NAME}/${STAGE} \
  --work-dir ${WORK_DIR} \
  --train_module.float8_config.enabled=true \
  --model.block.feed_forward.control_scope=gate \
  trainer.work_dir=${WORK_DIR}
