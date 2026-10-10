# Isolated synthetic distributed training efficiency

Prepared Dense/C0/C6 on one8GPU allocation, original models/optimizers/FP8 and global batch retained. Matched microbatch32768,5warmup+10measured optimizer steps per case. Source checkpoints untouched; no full training or quality comparison.

Submitted DENSE-C0-C6-TRAINING-STEP: `mose-eff-step-r1-p1-1x8-pool-1007-0406-44a9-70355`, pool 1x8, priority1; script `/mnt/shared-storage-gpfs2/intern-pretrain-shared02/shijiayang-p/archspace-MoSE/reproduce/olmo-core-backend/scripts/analysis/Training_Efficiency/run_benchmark.sh`.

04:22 HKT: Succeeded active0/succeeded1, all3separate SHA256 outputs and full recipe/arithmetic/finite metrics verified. Mean secondsDense3.690175,C02.901662,C62.987917; allocatedGiBDense51.269,C056.488,C666.305. C6 throughput1.235xDense but allocated1.293xDense. See RESULTS.md.
