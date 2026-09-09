---
name: mose-remote-gpu-submit
description: Prepare or submit Shijiayang's MoSE GPU jobs through the designated SSH submission host using gpu_run on the arch partition. Apply when generating submission commands or launching training/analysis jobs; keep code edits and ordinary work local.
---

# MoSE remote GPU submission

## Fixed submission context

- Local development repository: `/mnt/shared-storage-user/intern-pretrain-shared02/shijiayang-p/archspace-MoSE` (use the actual local workspace if its mount alias differs).
- SSH command supplied by the user: `ssh -CAXY cpu-develop.jitao+root.ailab-llmagent.ws@h.pjlab.org.cn`.
- Personal repository path visible on the submission host: `/mnt/shared-storage-gpfs2/intern-pretrain-shared02/shijiayang-p/archspace-MoSE/`.
- Activate the submission environment with **`conda activate MoSE`**, case-sensitive. This is the submission environment, not necessarily the training environment chosen inside a job's launcher.
- Submit through **`gpu_run arch NODES GPUS_PER_NODE JOB_NAME ABSOLUTE_SCRIPT_PATH`**. Only the **`arch`** partition is authorized. Apply the resource preferences below unless the user explicitly specifies otherwise.
- The local development machine cannot see the `arch` partition. This is expected: do not try to submit locally or switch partitions to work around it.

## Resource preferences

- Prefer whole 8-GPU nodes (`GPUS_PER_NODE=8`) for GPU jobs, including analysis, unless the user specifies a different allocation or a workload-specific exception is agreed. An 8-GPU allocation does not by itself make a single-process analysis script use all eight GPUs.
- For full training runs, prefer **2 nodes × 8 GPUs** (`gpu_run arch 2 8 ...`). Use **4 nodes × 8 GPUs** for acceleration only when the user explicitly requests that allocation for the current submission; do not carry an old 4×8 choice into later submissions by default.
- Changing the allocation does not authorize changing global batch size, token budget, scheduler, or other experimental hyperparameters. Check that the launcher's distributed settings and gradient accumulation support the requested allocation.
- When replacing a previously submitted job, establish that the old job has stopped before launching another job with the same checkpoint/output paths. If cancellation or status checks need additional remote authority under the shared-host boundary, ask the user to stop/confirm the old job or explicitly authorize those operations. Do not assume a new job name isolates training outputs.

## Shared-host boundary

The SSH host is a shared submission gateway, not a development or analysis machine.

- Edit code, write launchers, inspect files, run ordinary tests, and do Git work **locally**.
- On the shared host, perform only the requested SSH/session handling, `conda activate MoSE`, and the authorized `gpu_run arch ...` submission. Use absolute script paths; no remote directory exploration is needed.
- Do not remotely run Python, training, analysis, tests, package installation, cloning, pulling, editing, copying, file listings, environment dumps, resource probes, or scheduler queries. Do not execute/source the training launcher on the gateway. Its preparation and computation belong inside the scheduled job.
- Do not treat the SSH account's `root` name as permission to access or alter other users' files or global settings.
- If conda activation or gpu_run is unavailable, stop and ask the user for the required initialization steps; do not discover or modify the remote environment on your own.
- Remote monitoring, log retrieval, cancellation, or other operations require a new explicit instruction that permits the action. Reading accessible shared outputs locally remains allowed within the user's task.

## Workflow

1. Establish whether the user requested an actual submission or only scripts/commands. Creating this skill or a launcher is **not** authorization to launch a GPU job.
2. Prepare and validate the launcher locally. Map its repository-relative path onto the user's remote repository root above. Do not substitute another user's checkout. Do not assume a commit/push is required for shared storage, or claim remote visibility/synchronization has been verified without evidence. If the mount mapping is uncertain, ask rather than copying or editing remotely.
3. For an authorized submission, state the script, job name, node/GPU count, and `arch` partition. If the requested workload/resource count is unclear, clarify before launching. When only giving commands, include the SSH and conda steps, not a standalone local gpu_run command.
4. Connect using the supplied SSH command (interactive PTY if needed). Keep host-key verification enabled. Do not store passwords, tokens, or private keys in this skill or print credentials. Stop for user input if authentication or host identity needs confirmation.
5. Activate `MoSE`; submit only if activation succeeds. Use `&&` or check the activation result, so a failed activation cannot fall through to a submission in the wrong environment.
6. Capture the submission response/job ID and report exactly what was confirmed. A submitted job is not a completed job. If SSH disconnects or the response is ambiguous, **do not automatically resubmit**: the first submission may already exist. Ask the user to confirm its status before retrying.

## Command template

These are two stages: the first command is local; the activation and submission are entered in the resulting SSH session. Replace the uppercase workload placeholders before use.

```bash
ssh -CAXY cpu-develop.jitao+root.ailab-llmagent.ws@h.pjlab.org.cn
```

```bash
conda activate MoSE && gpu_run arch NODES GPUS_PER_NODE JOB_NAME /mnt/shared-storage-gpfs2/intern-pretrain-shared02/shijiayang-p/archspace-MoSE/RELATIVE_LAUNCHER_PATH
```

Keep this skill focused on submission location and permissions. For experiment hyperparameters, use the available `mose-experiment-defaults` skill when applicable; do not duplicate those defaults here. A user's later explicit change to the submission policy takes precedence, but quota/access failures alone never authorize a different host, partition, or workflow.
