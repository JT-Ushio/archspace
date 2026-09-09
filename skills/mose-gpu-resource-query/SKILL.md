---
name: mose-gpu-resource-query
description: Query agent or arch GPU scheduling availability with read-only rjob predict-only, especially whole 8-GPU nodes and 2x8 training allocations. Use for GPU capacity questions, not actual submission or cancellation.
---

# MoSE GPU resource query

## Scope

- Query when requested by the user; creating or reading this skill does not authorize a query. Use local `agent` and, when remote queries are authorized, the designated SSH host for `arch`. This narrow exception to the submission-only policy does not permit remote development, installation, training, or unrelated exploration.
- Always include **`--predict-only=true`**. Never remove it, add `--delete` or packaging/upload options, or submit a real job as a capacity probe. The verified local implementation sends a GET to the quota prediction endpoint and returns before job creation.
- For unfamiliar CLI versions, first check `rjob submit --help` for `--predict-only`. If unsupported, stop rather than guessing a submission option.
- Use normal CLI authentication without printing credentials. Keep SSH host-key verification enabled. Do not modify remote files or shell initialization; exit SSH after querying.

## Local agent query

Use GPU charged group `llmagent_gpu`, not the workspace CPU group. The verified local cluster entry was `https://h.pjlab.org.cn`. If sandbox networking prevents DNS/access, request normal network permission instead of treating failure as zero capacity.

```bash
rjob submit --predict-only=true --name=agent-gpu-capacity-check \
  --namespace=ailab-llmagent --charged-group=llmagent_gpu \
  --private-machine=group --gpu=8 --cpu=32 --memory=262144 -P 2 \
  --host-network=true \
  --custom-resources brainpp.cn/fuse=1 \
  --custom-resources rdma/mlnx_shared=8 \
  --custom-resources mellanox.com/mlnx_rdma=1 \
  --image=registry.h.pjlab.org.cn/ailab/ml-base:22.04-pjlab-20260323 \
  -- /bin/true
```

## Remote arch query

Connect locally:

```bash
ssh -CAXY cpu-develop.jitao+root.ailab-llmagent.ws@h.pjlab.org.cn
```

In the SSH session, activate `MoSE` (case-sensitive), and query only if activation succeeds:

```bash
conda activate MoSE && \
env -u http_proxy -u https_proxy -u no_proxy \
    -u HTTP_PROXY -u HTTPS_PROXY -u NO_PROXY \
    KUBEBRAIN_CLUSTER_ENTRY=http://wangyixiuan-cpu.linzhouhan.ailab-llmarchitecture.svc.pjlab.local:11451 \
    KUBEBRAIN_NAMESPACE=ailab-llmarchitecture \
rjob submit --predict-only=true --name=arch-gpu-capacity-check \
    --namespace=ailab-llmarchitecture --charged-group=llmarchitecture_gpu \
    --private-machine=group --gpu=8 --cpu=32 --memory=262144 -P 2 \
    --host-network=true \
    --custom-resources brainpp.cn/fuse=1 \
    --custom-resources rdma/mlnx_shared=8 \
    --custom-resources mellanox.com/mlnx_rdma=1 \
    --image=registry.h.pjlab.org.cn/ailab/ml-base:22.04-pjlab-20260323 \
    -- /bin/true
```

This endpoint and resource shape match the verified `gpu_run arch` workflow. Proxy removal is command-scoped: an initial prediction returned `response error: EOF`; the same query succeeded with these proxy variables removed. A continuing error means the query failed, not that GPUs are unavailable.

## Interpreting results

- `--gpu=8 -P 2` tests two whole eight-GPU nodes with 32 CPU cores and 256 GiB host memory per node plus the listed network/FUSE constraints. Failure is not necessarily GPU shortage alone.
- Change `-P 2` to `-P 1` to check a single whole eight-GPU node, particularly when the two-node probe reports only one schedulable replica.
- Report timestamp/timezone, partition, requested shape, schedulable replica count, and relevant candidate nodes. Predictions are instantaneous, do not reserve resources, and do not guarantee actual placement.
- The API only returns some candidate nodes. **Do not report displayed GPU sums or counts of eight-GPU rows as exact partition-wide totals.** Label any sum as a returned-list subtotal. Prefer the explicit schedulable replica count; failed-prediction tables need not enumerate every usable node.
- Exact totals or a complete list of idle eight-GPU nodes need an authorized complete node/quota inventory or platform resource view. State the limitation rather than guessing new API endpoints or broadening access.
- Scheduling feasibility is not GPU health. Flag candidates with known recent container/driver failures and seek health confirmation. Do not permanently blacklist nodes or assume a historical fault persists.
- The verified CLI may print a blank `created rjob_name:` after prediction and exit 0 after an API failure. Inspect the body: a blank ID is not a created job, and exit 0 alone does not establish query success.
- Queries do not authorize submission, cancellation, or rescheduling. For an explicitly requested real submission use `mose-remote-gpu-submit`.
