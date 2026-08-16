# Architecture: Cubit

## 1. Basic Information

| Item | Details |
|---|---|
| Architecture Name | Cubit: Token Mixer with Kernel Ridge Regression |
| Parent ARCH-PROP ID | [Issue #1](https://github.com/InternLM/archspace/issues/1) |
| Current ARCH-PROP ID | [Issue #20](https://github.com/InternLM/archspace/issues/20) |
| Paper | [Cubit: Token Mixer with Kernel Ridge Regression](https://arxiv.org/html/2605.06501) |

## 2. Current Implementation

The first implementation targets OLMo-core's OLMo3-1B training stack and follows the external
patch-and-recipe organization used by the MoSE branch. It does not modify an OLMo-core checkout:

- `reproduce/olmo-core-backend/src/olmo_cubit/` contains the Cubit module, serializable config,
  optimizer compatibility shim, and non-mutating `TransformerConfig` patch.
- `reproduce/olmo-core-backend/cfgs/` contains matched OLMo3-1B baseline and Cubit recipes.
- `reproduce/olmo-core-backend/tests/` verifies the paper formula, causal and packed-document
  masks, gradients, model initialization, config round trips, and recipe construction on CPU.

The KRR matrix construction and triangular solve are intentionally written as readable FP32
PyTorch in this correctness-first version. The final attention aggregation uses OLMo's configured
backend; the server recipe selects FlashAttention 3.

See [`reproduce/olmo-core-backend/README.md`](reproduce/olmo-core-backend/README.md) for exact
installation, verification, training, and override commands.

## 3. Status and Scope

Implemented:

- OLMo3-1B baseline and Cubit recipes with identical data, optimizer, batch, evaluation, and W&B
  settings.
- Independent reference projection (paper default) and shared-key ablation.
- Limited-Range Rescale (LRR), learnable reference scale, and positive learned ridge coefficient.
- Causal, sliding-window, and packed-document masks.
- Half-open Cubit layer selection `[cubit_start_layer, cubit_end_layer)`.
- HSDP training with BF16 parameters/reductions and FP32 KRR math.
- FlashAttention 3 for the final `A @ solution` aggregation on supported GPUs.

Deferred to later optimization/conversion work:

- A fused or blockwise KRR kernel that removes the dense score matrix.
- Tensor/context parallelism and KV-cached autoregressive decoding.
- A Hugging Face architecture and OLMo-to-HF checkpoint converter under `archs/`.
