# C0–C7 matched-budget comparison

All eight runs completed 14,306 optimizer steps of the same 30B-token FP8 recipe. Each has 24,117,248 projection parameters per Transformer block, matching the fully unshared rank-512 MoSE reference. The primary metric is mean `train/CE loss` over steps 13,307–14,306; lower is better. Validation is an equal-weight mean over the same 11 `eval/lm/*-validation/CE loss` sources at step 14,000, the last shared validation point.

| Run | Configuration relative to rank-512 reference | Full Train CE | Last 1,000 Train CE | Δ vs C0 | Validation macro CE |
| --- | --- | ---: | ---: | ---: | ---: |
| C0 | K SiLU, Down RMSNorm; no sharing | 2.793340 | 2.479042 | 0 | 2.740454 |
| C1 | C0 + KV sharing | 2.797321 | 2.481070 | +0.002029 | 2.745497 |
| C2 | C0 + shared Gate/Up | 2.791325 | 2.475732 | −0.003310 | **2.734792** |
| C3 | C0 + KV and Gate/Up sharing | 2.796868 | 2.478402 | −0.000640 | 2.740359 |
| C4 | Q/K/V/O SiLU + KV sharing; Gate/Up unshared | 2.798778 | 2.479408 | +0.000366 | 2.738418 |
| C5 | Q SiLU, K/V/O Identity + shared Gate/Up | 2.791086 | 2.475784 | −0.003257 | 2.736490 |
| **C6** | **Q/K SiLU, V/O Identity + shared Gate/Up** | **2.789434** | **2.474827** | **−0.004215** | 2.736803 |
| C7 | C6 + SiLU between both shared Gate/Up factors | 2.808255 | 2.492346 | +0.013304 | 2.754680 |

Every run uses Down RMSNorm without learnable gain. C0–C6 use Identity between the Gate/Up low-rank factors while retaining the usual SwiGLU operation; C7 changes those two inter-factor operations to SiLU. Q/K/V/O activations and exact rank reallocations are in the corresponding config and launcher files.

C6 has the best matched final-1,000-step Train CE, but its gain over C2 is only 0.000905 and is from one run per configuration. C2 has the best last validation macro CE. Adding Gate/Up SiLU in C7 worsens final-1,000-step Train CE by 0.017519 versus the otherwise matched C6, and worsens final validation macro CE by 0.017877. The C7–C6 Train CE gap is positive in every 1,000-step block, so this is not a single-point artifact. Within this sweep, Gate/Up sharing helps the K-SiLU control (C2 vs C0), while KV sharing does not (C1 vs C0, C3 vs C2).

The raw offline W&B records contain exactly one `train/CE loss` value at every optimizer step 1–14,306 for each run. Three values are zero and are treated as logging anomalies: C3 step 14,304, C6 step 12,516, and C7 step 12,516. For an exactly matched comparison, steps 12,516 and 14,304 are excluded from **all eight** runs. Full-run means therefore use 14,304 common steps; final-1,000-step means use 999 common steps. The previously reported C6 last-1,000 mean of 2.474763 excluded only step 12,516 across C2/C5/C6; the value here differs by 0.000064 because C3's step 14,304 is now excluded across all eight.

W&B runs: [C0](https://wandb.ai/archspace/MoSE/runs/olmo3-1b-r512matched-c0-k-silu-down-rms-fp8-30b), [C1](https://wandb.ai/archspace/MoSE/runs/olmo3-1b-r512matched-c1-k-silu-down-rms-kv-fp8-30b), [C2](https://wandb.ai/archspace/MoSE/runs/olmo3-1b-r512matched-c2-k-silu-down-rms-gateup-fp8-30b), [C3](https://wandb.ai/archspace/MoSE/runs/olmo3-1b-r512matched-c3-k-silu-down-rms-kv-gateup-fp8-30b), [C4](https://wandb.ai/archspace/MoSE/runs/olmo3-1b-r512matched-c4-attention-silu-down-rms-kv-fp8-30b), [C5](https://wandb.ai/archspace/MoSE/runs/olmo3-1b-r512matched-c5-q-silu-down-rms-gateup-fp8-30b), [C6](https://wandb.ai/archspace/MoSE/runs/olmo3-1b-r512matched-c6-qk-silu-down-rms-gateup-fp8-30b), [C7](https://wandb.ai/archspace/MoSE/runs/olmo3-1b-r512matched-c7-qk-silu-ug-silu-down-rms-gateup-fp8-30b).
