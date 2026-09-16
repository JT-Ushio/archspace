# W&B synchronization — 2026-09-10

User requested `conda activate MoSE` then `wandb_sjy <log-directory>`, followed by continued monitoring of the four experiments. Scope used: every executed attempt of the Q/K/V/O single-SiLU ablations, not unrelated experiments.

Log root on submission host:
`/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb`

Each exact segment was identified from the corresponding replica's W&B startup
log. Completion below is based on `/tmp/wandb_sjy.log`, which is the output file
used by the submission-host `wandb_sjy` wrapper. No credentials were read or
stored.

| Variant | Attempt | Directory under root | Sync status |
| --- | --- | --- | --- |
| Q | original | offline-run-20260909_225150-olmo3-1b-all-linear-q-only-silu-fp8-30b | Finished syncing (beta) |
| Q | r1 | offline-run-20260910_025421-olmo3-1b-all-linear-q-only-silu-fp8-30b | Finished syncing (beta) |
| Q | r2 | offline-run-20260910_135149-olmo3-1b-all-linear-q-only-silu-fp8-30b | Finished syncing (beta) |
| Q | r3 | offline-run-20260910_164238-olmo3-1b-all-linear-q-only-silu-fp8-30b | Stopped18:17:50; sync PID4069624; Finished syncing (beta) verified18:19:57 |
| Q | r4 | offline-run-20260910_184414-olmo3-1b-all-linear-q-only-silu-fp8-30b | Completed14306/Succeeded; sync PID4103356; Finished syncing (beta) verified approximately20:19 |
| K | original | offline-run-20260909_234605-olmo3-1b-all-linear-k-only-silu-fp8-30b | Finished syncing (beta) |
| K | r2 | offline-run-20260910_034208-olmo3-1b-all-linear-k-only-silu-fp8-30b | Finished syncing (beta) |
| K | r3 | offline-run-20260910_135352-olmo3-1b-all-linear-k-only-silu-fp8-30b | Beta reached the truncated tail and reported unexpected EOF; legacy retry returned done |
| K | r4 | offline-run-20260910_164714-olmo3-1b-all-linear-k-only-silu-fp8-30b | Completed14306/Succeeded; sync PID4070268; Finished syncing (beta) verified18:21 |
| V | original | offline-run-20260910_000558-olmo3-1b-all-linear-v-only-silu-fp8-30b | Beta reported unexpected EOF; legacy retry returned done |
| V | r2 | offline-run-20260910_034546-olmo3-1b-all-linear-v-only-silu-fp8-30b | Beta reached the truncated tail and reported unexpected EOF; legacy retry returned done |
| V | r3 | offline-run-20260910_135411-olmo3-1b-all-linear-v-only-silu-fp8-30b | Legacy sync returned done |
| V | r4 | offline-run-20260910_170803-olmo3-1b-all-linear-v-only-silu-fp8-30b | Completed14306/Succeeded; sync PID4076681; Finished syncing (beta) verified18:44 |
| O | original | offline-run-20260910_001011-olmo3-1b-all-linear-o-only-silu-fp8-30b | Finished syncing (beta) |
| O | r2 | offline-run-20260910_034544-olmo3-1b-all-linear-o-only-silu-fp8-30b | Finished syncing (beta) |
| O | r3 | — | No segment: job stopped from Inqueue and never started training |
| O | r4 | offline-run-20260910_163432-olmo3-1b-all-linear-o-only-silu-fp8-30b | Stopped18:17:10; sync PID4069336; Finished syncing (beta) verified18:18:49 |
| O | r5 | offline-run-20260910_182247-olmo3-1b-all-linear-o-only-silu-fp8-30b | Completed14306/Succeeded; sync PID4083937; Finished syncing (beta) verified19:11 |

## Verification notes

`wandb_sjy` runs `wandb sync` in the background and appends output to
`/tmp/wandb_sjy.log`. Abruptly terminated training can leave a transaction log
without a clean tail. The beta uploader recovered history before returning
`unexpected EOF`; the W&B-advertised `--legacy` mode then completed those exact
segments. Segments sharing one run ID were serialized.

All known training-produced segments through Q-r4/K-r4/V-r4/O-r5 have been
submitted and verified as above. All four completed full training (14306 steps)
with final checkpoint saves and successful exits. Final synchronization output
was rechecked approximately20:19; no known segment remains pending.
For comparisons, use explicit training step/tokens and handle replayed steps
after checkpoint resumes rather than assuming W&B history is deduplicated.
