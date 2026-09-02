## Notes

💡 To make your Proposal easier to validate and accept, provide implementation code that is **reproducible**, **runnable**, and **easy to use**, together with **clear and complete documentation**.

💡 The model architecture code must be converted to a Hugging Face Transformers-compatible format and placed in the `archs/` directory. **Only changes within `archs/` will be merged**.

💡 **All reproduction code must be placed in the `reproduce/` directory**, which you may also use as your working directory during development.

# Architecture: `<Architecture Name>`

---

## 1. Basic Information

| Item                 |                          Details                          |
| -------------------- | :-------------------------------------------------------: |
| Architecture Name    |                   `<Architecture Name>`                   |
| Parent ARCH-PROP ID  | [Issue \#N](https://github.com/InternLM/archspace/issues) |
| Current ARCH-PROP ID | [Issue \#N](https://github.com/InternLM/archspace/issues) |

## 2. Reproducing the Experiments

### 2.1 Environment Setup

> Specify the required hardware and software environment, and provide complete installation instructions. Pin key dependency versions to ensure the environment can be reproduced reliably.

### 2.2 Data Preparation

> Architecture experiments should generally use the same data as the baseline. If the data or data-processing pipeline differs from the baseline, describe the data source and the complete preparation process here.

### 2.3 Training Pipeline

> Provide all training scripts, configuration files, and commands required to reproduce the training process. The commands should run without requiring modifications to the source code.

### 2.4 Evaluation Pipeline

> Provide all evaluation scripts, configuration files, and commands required to reproduce the reported results. Clearly specify the evaluation metrics and expected outputs.

### 2.5 Model Weights and Experiment Logs

> 1. Use [Weights & Biases](https://wandb.ai/site/) to record training logs.
> 2. After completing the validation experiments, convert the model architecture code to a [Hugging Face Transformers-compatible format](https://huggingface.co/docs/transformers/v5.14.0/en/main_classes/model).## Notes
>    💡 To make your Proposal easier to validate and accept, provide implementation code that is **reproducible**, **runnable**, and **easy to use**, together with **clear and complete documentation**.
>    💡 The model architecture code must be converted to a Hugging Face Transformers-compatible format and placed in the `archs/` directory. **Only changes within `archs/` will be merged**.
>    💡 **All reproduction code must be placed in the `reproduce/` directory**, which you may also use as your working directory during development.

# Architecture: `<Architecture Name>`

---

## 1. Basic Information

| Item                 |                          Details                          |
| -------------------- | :-------------------------------------------------------: |
| Architecture Name    |                   `<Architecture Name>`                   |
| Parent ARCH-PROP ID  | [Issue \#N](https://github.com/InternLM/archspace/issues) |
| Current ARCH-PROP ID | [Issue \#N](https://github.com/InternLM/archspace/issues) |

## 2. Reproducing the Experiments

### 2.1 Environment Setup

> Specify the required hardware and software environment, and provide complete installation instructions. Pin key dependency versions to ensure the environment can be reproduced reliably.

### 2.2 Data Preparation

> Architecture experiments should generally use the same data as the baseline. If the data or data-processing pipeline differs from the baseline, describe the data source and the complete preparation process here.

### 2.3 Training Pipeline

> Provide all training scripts, configuration files, and commands required to reproduce the training process. The commands should run without requiring modifications to the source code.

### 2.4 Evaluation Pipeline

> Provide all evaluation scripts, configuration files, and commands required to reproduce the reported results. Clearly specify the evaluation metrics and expected outputs.

### 2.5 Model Weights and Experiment Logs

> 1. Use [Weights & Biases](https://wandb.ai/site/) to record training logs.
> 2. After completing the validation experiments, convert the model architecture code to a [Hugging Face Transformers-compatible format](https://huggingface.co/docs/transformers/v5.14.0/en/main_classes/model).
