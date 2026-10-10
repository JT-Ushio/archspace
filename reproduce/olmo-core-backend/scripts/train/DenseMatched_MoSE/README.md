# Dense-parameter-matched MoSE comparison

This directory defines the two dense-parameter-matched MoSE runs. It intentionally
does not add or submit D0. The native dense OLMo3-1B entry point already exists
under cfps/BS256_Seq8K_30B_WSD_FP8/OLMo3-1B-DenseBaseline.py.

## Architecture

Both runs keep the primary SwiGLU operation, QK norms, attention softmax and
outer Transformer norms. Attention bottlenecks use Identity. Gate/Up
bottlenecks use SiLU. Down uses RMSNorm without a learnable gain.

| Run | Q/O rank | K/V | Gate/Up | Down |
| --- | ---: | --- | --- | --- |
| F0 | 1024 | independent, rank 1024 | independent, rank 1640 | rank 1632, RMSNorm |
| F1 | 1024 | shared input, rank 1360 | shared input, rank 1824 | rank 1632, RMSNorm |

F0 isolates the effect of the low-dimensional nonlinear parameterization. F1
tests whether reallocating the same parameter budget to shared K/V and
Gate/Up latents improves over F0.

## Parameter budget

For OLMo3-1B, d_model is 2048, FFN width is 8192, and there are 16 blocks.
Both F0 and F1 contain exactly 32,768 fewer projection parameters
per block than the native dense Q/K/V/O and SwiGLU projections:

    dense OLMo3-1B total:       1,484,916,736
    F0/F1 expected total:       1,484,392,448
    difference:                      -524,288  (-0.03531%)

The config validates the merged ranks and sharing scopes before model
allocation. A launcher override that breaks this budget fails immediately.

## Fixed training recipe

- global batch: 256 sequences / 2,097,152 tokens
- sequence length: 8192
- rank microbatch: 32,768 tokens
- budget: 30B tokens / about 14,306 optimizer steps
- Muon LR: 0.005
- WSD: 2,000-step warmup, final 20% decay
- Float8 conversion enabled
- checkpoint every 1,788 steps; retain all checkpoints

The launchers repeat all architecture settings as command-line safeguards:

    scripts/train/DenseMatched_MoSE/run_f0_unshared.sh
    scripts/train/DenseMatched_MoSE/run_f1_kv_gateup_shared.sh

They prepare commands only; no GPU job is submitted by these files.

## Comparison

Compare actual matched optimizer steps/tokens. Primary metrics are the last
1,000-step train CE and the equal-weight macro CE over the same 11 validation
sources at the last common validation step. Also report TPS/device, MFU and
peak memory because parameter matching does not guarantee equal kernel
efficiency.

## 30B-token results

All three runs reached step 14306. The train-window numbers below average all
1,000 W&B history values from steps 13307 through 14306. Validation is the
equal-weight macro CE over the same 11 LM validation sources at step 14000.
TPS/device and MFU use console samples from the same final 1,000-step window.

| Run | Parameters | Train CE | Validation macro CE | TPS/device | MFU |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dense | 1,484,916,736 | 2.35759 | 2.62622 | 69,396 | 67.95 |
| F0 | 1,484,392,448 | 2.39244 | 2.65557 | 56,848 | 55.65 |
| F1 | 1,484,392,448 | 2.39739 | 2.65980 | 62,794 | 61.47 |

Relative to dense, F0 is +0.03485 train CE and +0.02935 validation macro CE;
F1 is +0.03981 and +0.03358, respectively. Dense has lower validation CE on
all 11 sources. Across every common 500-step validation from step 4000 through
14000, dense also has the lowest macro CE. F1 is +0.00495 train CE and
+0.00423 validation macro CE relative to F0; from step 7000 onward F0 has the
lower validation macro CE at every evaluation.

F1 recovers part of F0's execution cost: it is about 10.5% faster than F0 in
the final window. It remains about 9.5% slower than dense, while F0 is about
18.1% slower than dense. No peak-memory metric was recorded by these runs.

The direct F0/F1 comparison is clean: both use commit
`a3e4964fb0b42fcdaffb52854e09705cd1d2ec5a`, rank microbatch 32768, and the
same data, seeds, global batch, optimizer, schedule and FP8 settings. The dense
reference is older: commit `7ef1da99f683e77e2abc8e308868077419b7c352`
and rank microbatch 65536. Its global batch, sequence length, data seed, init
seed, optimizer, schedule, token budget and FP8 setting match. Therefore the
F0/F1 result supports a firm architecture conclusion, while the dense
comparison should be described as strong evidence with a code-version and
microbatch-size caveat.
