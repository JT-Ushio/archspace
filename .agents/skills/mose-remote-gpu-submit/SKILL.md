---
name: mose-remote-gpu-submit
description: Prepare or submit Shijiayang's MoSE GPU jobs through the designated SSH host using gpu_run on arch, and sync stopped attempts with wandb_sjy before authorized restarts. Apply to submission commands and job restart workflows; keep development local.
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
- On the shared host, perform only the requested SSH/session handling, `conda activate MoSE`, authorized `gpu_run arch ...` submissions, and the `wandb_sjy` synchronization described below. Use absolute paths; no remote directory exploration is needed.
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

## Stopped attempt: sync first, then restart

User preference: for a task being monitored with restart authorization, as soon
as it is confirmed stopped/killed, submit that attempt's W&B logs before
submitting its replacement. Do not defer synchronization until all experiments
finish. This does not authorize restarting a task the user explicitly paused or
cancelled, or bypassing a scheduler/access prohibition.

1. Confirm the old attempt is terminal and has zero active replicas using
   authorized status checks. Queueing, a log timeout, or an SSH disconnect is
   not evidence that training stopped; do not duplicate a queued/live job.
2. Resolve the exact offline-run directory for that attempt from its training
   logs or an existing verified ledger. The argument is the W&B offline-run
   directory, not a scheduler log file or the parent directory of every run.
   If the attempt never started training and produced no W&B segment, record
   that fact and skip synchronization. If the path is unknown, resolve it with
   authorized log retrieval rather than guessing.
3. On the designated SSH gateway, immediately invoke:

   ```bash
   conda activate MoSE && wandb_sjy '/absolute/path/to/offline-run-...'
   ```

   Record the old job ID, exact segment path, invocation result, and any
   background PID/output location. A detached PID means synchronization was
   launched, not that upload completed. Do not re-upload a segment merely
   because completion has not yet been verified. If invocation fails or its
   result is ambiguous, report it and resolve that state before retrying.
4. After synchronization has been launched, submit the authorized replacement
   with `gpu_run arch`, normally 2×8, preserving the experiment configuration,
   RUN_NAME, stage, checkpoint directory, and resume behavior. Use a unique
   scheduler job name and capture its actual ID. A background upload need not
   block training restart; keep its completion explicitly pending.
5. Verify upload completion through authorized output/log checks when available.
   Serialize uploads of different segments sharing a W&B run ID until the
   previous upload finishes; do not invent merge/append flags or claim histories
   merged without evidence. This sync exception does not authorize remote
   wrapper inspection, filesystem exploration, installations, or credential
   reads; request any additional access needed for verification. Never print
   or persist W&B credentials.

Keep this skill focused on submission location and permissions. For experiment hyperparameters, use the available `mose-experiment-defaults` skill when applicable; do not duplicate those defaults here. A user's later explicit change to the submission policy takes precedence, but quota/access failures alone never authorize a different host, partition, or workflow.
