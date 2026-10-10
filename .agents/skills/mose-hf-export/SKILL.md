---
name: mose-hf-export
description: Convert MoSE/OLMo-core checkpoints to Hugging Face with the shared exporter, or maintain HF compatibility when adding or changing model architectures in archspace-MoSE. Includes forward-contract and optional numerical validation.
---

# MoSE HF export and architecture compatibility

Use the shared implementation under `reproduce/olmo-core-backend/src/olmo_mose/hf_export/`. Read [the conversion guide](../../../reproduce/olmo-core-backend/scripts/convert/README.md) for commands, format, supported families and forward differences. The same `convert`, `batch`, `inspect`, `audit-config`, and `validate` entry points apply to every supported experiment.

## Adding or changing an architecture

Every new architecture must include a corresponding HF adapter module in `hf_export/architectures/`, imported by its `__init__.py`. Register exact native classes; a new subclass must not silently inherit Dense export. Existing rank, width or activation combinations reuse the appropriate family adapter. Do not create a separate converter per experiment.

Describe the actual built model, including shapes, activations, norm placement/eps/gain, attention scale, RoPE layout, sharing and learned parameters. Cover every persistent state entry. New forward arithmetic requires an equivalent operator in standalone `runtime_ops.py`; exported runtime must depend only on PyTorch/Transformers, not training imports. Do not merge nonlinear projections into Dense weights or substitute initialization constants for trained gates/controllers.

Add small-model regression cases for checkpoint→HF reload, forward logits/intermediates, parameter aliases and cached decoding. Audit saved-config serialization and meta-device state keys/shapes/counts. Update the guide's support table. Unknown classes, state keys, incomplete checkpoints or unsupported semantics must fail explicitly.

## Converting checkpoints

Use saved `config.json` and actual checkpoint metadata. Never infer ranks/sharing from run names. Preserve sources, reject nonempty output directories, and do not fall back from an incomplete newest step. DCP export reads model tensors only. Check alias values before storing one shared tensor; restore shared Parameter identities on HF load. Preserve special token IDs and padded embedding rows.

Use the batch JSON manifest for multiple checkpoints and retain per-item success/failure results. No uploading or job submission is implied by a local conversion request.

## Validation and reporting

Distinguish structural audit, export completion and numerical verification. The user can defer numerical verification; still write the verification code and report `not_run`. Do not run numerical forward/checkpoint validation if the user has asked to defer it.

For an authorized numerical check, use `validate` to compare reloaded weights, identical unpadded logits, per-layer outputs, matching shifted-target CE, HF labels, external embeddings, cached decode and padding. Native `input_embeddings` maps to HF `inputs_embeds`; native labels are already shifted, HF labels are not. Packed-document/CP arguments and training z-loss do not have interchangeable HF semantics. Record dtype, tolerance and the non-FP8 eager master-weight reference; structural compatibility alone is not evidence of numerical equivalence.
