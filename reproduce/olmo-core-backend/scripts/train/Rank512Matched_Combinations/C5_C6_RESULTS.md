# Q and QK SiLU with Gate/Up sharing: matched train CE

C5 and C6 match the all-rank-512, unshared projection budget exactly at 24,117,248 parameters per block. Both use Gate/Up sharing (rank 568), Down RMSNorm (rank 520), Q/K/V rank 512, O rank 496, no attention sharing, and the same 30B-token FP8 recipe. C5 applies SiLU only to Q; C6 applies SiLU to Q and K. C2 applies SiLU only to K with the same ranks and sharing.

Both new jobs completed all 14,306 optimizer steps and saved the final checkpoint. Their W&B segments were synchronized. The offline histories have complete step coverage. At C6 step 12516, `train/CE loss=0`, `train/PPL=1`, `train/Z loss=0`, and `optim/total grad norm=0.01919`; this is an anomalous logged row. All values below exclude that same step from C2, C5, and C6, leaving 14,305 aligned steps.

| Run | Q/K activation | Full mean CE | Last 1000 CE | Last 500 CE | Last 100 CE |
| --- | --- | ---: | ---: | ---: | ---: |
| C2 | Linear / SiLU | 2.791299 | 2.475668 | 2.465522 | 2.457145 |
| C5 | SiLU / Linear | 2.791060 | 2.475720 | 2.465583 | 2.457223 |
| **C6** | **SiLU / SiLU** | **2.789407** | **2.474763** | **2.464741** | **2.456443** |

C6 improves the primary last-1000-step train CE over C2 by 0.000905. C5 has almost the same late CE as C2, so replacing K SiLU with Q SiLU alone does not show a useful gain. C6 is better in each 1000-step block from step 1001 through 14306, but this is one seed and a small effect. The result does not close the roughly 0.12 CE gap to the dense baseline.

W&B runs:

- [C2](https://wandb.ai/archspace/MoSE/runs/olmo3-1b-r512matched-c2-k-silu-down-rms-gateup-fp8-30b)
- [C5](https://wandb.ai/archspace/MoSE/runs/olmo3-1b-r512matched-c5-q-silu-down-rms-gateup-fp8-30b)
- [C6](https://wandb.ai/archspace/MoSE/runs/olmo3-1b-r512matched-c6-qk-silu-down-rms-gateup-fp8-30b)
