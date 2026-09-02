# OLMo-core backend scripts

The new scripts are grouped by responsibility:

- `prepare/`: environment variables and conda activation.
- `train/`: executable training entrypoints used locally or as an RJob command.

## Gate-SiLU / Up-SiLU / Down-RMSNorm, all r2=512

This experiment uses:

```text
r1=0
r2=512
down_r1=0
down_r2=512
gate_nonlinearity=silu
up_nonlinearity=silu
down_nonlinearity=rms_norm
control=standard
control_scope=none
FP8=true
global_batch_size=256 sequences (2,097,152 tokens at 8K)
sequence_length=8192
max_duration=30,000,000,000 tokens
Muon LR=5e-3
schedule=WSD (2,000-step warmup, 20% linear decay)
checkpoint_interval=3,576 steps (~7.499B tokens)
max_checkpoints=None
```

The rank values and nonlinearities above are also fixed in the dedicated
configuration file; the training command repeats them as explicit safeguards.
There is no clipping or SiTU channel control in this experiment.

The dedicated configuration is isolated under:

```text
cfps/mose_r1_0_r2_512_down_r1_0_down_r2_512_gate_silu_up_silu_down_rmsnorm/
```

The per-rank microbatch is 65,536 tokens (8 sequences at 8K). With 32 GPUs
this is one microbatch per global step; with 16 GPUs OLMo-core accumulates two
microbatches to retain the same global batch of 256 sequences.

The training entrypoint sources `prepare/prepare_olmo_gpfs2.sh` itself.  Its
RJob command path is:

```text
/mnt/shared-storage-gpfs2/intern-pretrain-shared02/shijiayang-p/archspace-MoSE/reproduce/olmo-core-backend/scripts/train/run_OLMo3_1B_MoSE_r1_0_r2_512_down_r1_0_down_r2_512_GateSiLU_UpSiLU_DownRMSNorm_BS256_Seq8K_30B_WSD_FP8.sh
```

To activate the same environment in an interactive shell:

```bash
source /mnt/shared-storage-gpfs2/intern-pretrain-shared02/shijiayang-p/archspace-MoSE/reproduce/olmo-core-backend/scripts/prepare/prepare_olmo_gpfs2.sh
```
