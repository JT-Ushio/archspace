# Architecture: AHN

AHN evaluates linear-attention replacements in the OLMo3-1B architecture. It supports Gated
DeltaNet (GDN), Kimi Delta Attention (KDA), and Gated DeltaNet 2 (GDN2) with two layer layouts:

- linear attention in all 16 transformer layers;
- linear attention in the 12 layers that use sliding-window attention in OLMo3-1B, while layers
  3, 7, 11, and 15 retain the original Full Attention implementation.

## Basic Information

| Item | Details |
| --- | --- |
| Architecture Name | AHN |
| Parent ARCH-PROP ID | To be added |
| Current ARCH-PROP ID | To be added |

## Reproducing the Experiments

The complete OLMo-core extension, pinned dependencies, model patching, recipes, local checks,
server installation, CUDA validation, and training commands are documented in
[`reproduce/olmo-core-backend/README.md`](reproduce/olmo-core-backend/README.md).

The experiments use the OLMo3-1B stage-1 tokenizer, OLMo 150B-sample data mix, HSDP/Muon recipe,
and evaluation callbacks from the baseline. The architecture variants change only the sequence
mixers and the Muon handling required for their depthwise convolution weights.

## Model Weights and Experiment Logs

- Weights & Biases runs: To be added after server training.
- Model weights: To be added after validation.
- Hugging Face architecture and checkpoint converter: To be implemented after the OLMo-core
  experiments are validated.
