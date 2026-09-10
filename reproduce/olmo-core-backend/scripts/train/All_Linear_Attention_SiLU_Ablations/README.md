# All-linear: single attention SiLU ablations

Only one low-rank internal activation changes relative to
`All_Linear_Ablation/all_linear_fp8.py`. There are four independent experiments.

| Launcher | Q | K | V | O | Gate/Up/Down internal |
| --- | --- | --- | --- | --- | --- |
| run_q_only_silu_fp8.sh | SiLU | Identity | Identity | Identity | Identity |
| run_k_only_silu_fp8.sh | Identity | SiLU | Identity | Identity | Identity |
| run_v_only_silu_fp8.sh | Identity | Identity | SiLU | Identity | Identity |
| run_o_only_silu_fp8.sh | Identity | Identity | Identity | SiLU | Identity |

Primary SwiGLU `SiLU(gate) * up`, attention softmax, QK norms and outer RMSNorm
are unchanged. No model implementation or existing experiment is modified.
All factors retain rank512, parameter names/shapes and initialization. No
Gate/Up or QKV sharing; FFN r1/down_r1=0, r2/down_r2=512.

## Matched training settings

- Muon LR0.005, weight decay0.033, betas(0.9,0.95), grad clip1, z-loss1e-5.
- FP8 enabled, existing HSDP BF16 parameter / FP32 reduction and compilation.
- Global batch256, sequence8192: 2,097,152 tokens/optimizer step.
- Per-rank microbatch32768 tokens = 4 sequences; arch2x8 accumulates4 microsteps.
- Budget30B tokens; WSD2000-step warmup and final20% decay, unchanged decay shape.
- Checkpoints every1788 steps (approximately12.5% of budget), no periodic retention
  limit. Final/startup saves can add checkpoints.
- All other defaults (including initialization/data seeds and evaluations) inherited
  directly from all-linear. Tests compare the complete configs, allowing only the
  selected activation and W&B group to differ.

## Submission and restart

Through the designated SSH gateway, after `conda activate MoSE`, use
`gpu_run arch 2 8 JOB_NAME ABSOLUTE_LAUNCHER_PATH`.
Default run IDs: `olmo3-1b-all-linear-{q,k,v,o}-only-silu-fp8-30b`;
W&B group: `all-linear-attention-single-silu-fp8-30b`.
Each run has its own `${CKPT_DIR}/${RUN_NAME}/stage1` directory.
Ports29646–29649 are distinct.

The user authorizes concurrent submission, queueing, status/resource/log checks,
and replacement of killed jobs. Verify a job is terminal before replacing it;
query failures or an SSH disconnect alone do not establish termination. Never
overlap attempts writing the same run directory. Keep RUN_NAME/STAGE unchanged
on restart; use a new scheduler job name. Check capacity when deciding when to
replace a terminated attempt. Queueing is allowed.

OLMo's entrypoint loads the latest discoverable checkpoint from save_folder,
including optimizer/trainer/data-loader/RNG state. It cannot recover work since
the last valid checkpoint. A partially written checkpoint must be diagnosed,
not deleted or silently bypassed. Inspect restoration logs for each restart.

Preparation currently sets WANDB_MODE=offline. Offline restarts produce separate
log segments with the same run ID; collect all segments and align/deduplicate by
step when comparing. WANDB_RESUME=allow applies if online mode is enabled later;
it does not make offline segments one continuous local log.

Monitor until all four experiments finish the30B budget and final checkpoints
are verified. Scheduler creation/Running status is not proof of training completion.
User-requested stops and platform policy/quota denials must not be bypassed.

## Tests

Run `test_all_linear_ablation` and `test_single_attention_silu_ablation` using
the OLMo environment with backend src/tests and OLMo-core src on PYTHONPATH.
Tests cover full-config equivalence, launcher defaults, shapes/initialization,
projection forward/backward, primary SwiGLU preservation, and the2x8 batch size.
