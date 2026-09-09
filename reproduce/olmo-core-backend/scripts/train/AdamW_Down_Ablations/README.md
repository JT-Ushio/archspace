# AdamW × Down 激活消融

新增两组，不修改已有 Muon 实验。复用 `Down_Identity_Ablations` 的 30B 配方，替换整个优化器为 `olmo_core.optim.AdamWConfig`（实际 `torch.optim.AdamW`），而非仅修改 Muon 的 LR。

| 实验 | 优化器 | LR | Down 中间激活 | 启动脚本 |
| --- | --- | --- | --- | --- |
| AdamW-S | AdamW | 1e-3 | SiLU | `run_down_silu_adamw_fp8.sh` |
| AdamW-R | AdamW | 1e-3 | RMSNorm，无可学习 gain | `run_down_rms_adamw_fp8.sh` |

其余所有可选低秩中间激活（Q/K/V/O、Gate/Up）都是 SiLU。保留 OLMo3 原有 block、QK 及 FFN 输出 RMSNorm；“其余都是 SiLU”不表示替换这些原生归一化层。

## 固定参数

- 所有低秩投影 rank=512；`r1=down_r1=0`，`r2=down_r2=512`。
- Gate/Up 不共享；Q/K/V/O 不共享。
- 全局 batch=256 条序列，sequence length=8192，每步 2,097,152 tokens。
- max tokens=30,000,000,000，约 14,306 optimizer steps。
- WSD：2000 optimizer steps warmup，最后 20% 线性 decay，沿用现有 WSD 类。
- FP8 enabled，其他算子保留原 mixed-precision 策略。
- microbatch=32768 tokens/卡，梯度累积保持全局 batch 不变。
- 每 1788 steps 保存 checkpoint，约 12.5%；不删除周期 checkpoint。初始/结束保存可能额外增加数量。
- AdamW betas=(0.9, 0.95)，eps=1e-8，weight_decay=0.033。
- 与原 Muon 配方一致：embedding 和向量参数（本模型中的 norm weights）weight_decay=0；内部矩阵和 LM head 矩阵 weight_decay=0.033。所有参数均走 AdamW，LR=1e-3。
- 数据顺序、初始化 seed、gradient clipping、z-loss、评测频率、compile、FSDP 等继承相同 30B baseline。

无衰减分组使用当前无 bias 模型的 `embeddings.*`、`*norm*.weight` 名称匹配。测试会对实际 meta model 的所有参数逐一核对，确保和 Muon 原有 embedding/vector 分类一致；若以后新增 bias 或其他向量参数，应同步调整分组。

## 提交和命名

通过原有 gpu_run 包装器提交本目录内两个脚本。默认 run_name / 保存目录与 Muon 分开：

- `olmo3-1b-down-activation-silu-adamw-fp8-30b`
- `olmo3-1b-down-activation-rms-adamw-fp8-30b`

重跑请设置新的 `RUN_NAME`，避免复用 checkpoint。不要加载 Muon 的 optimizer state 作为 AdamW 的起点；这两组从头训练。

## 如何分析

优先比较相同步数/训练 tokens 的 validation CE/PPL、各验证集差值及末尾 train CE 均值：

`Delta_AdamW = loss(AdamW-SiLU) - loss(AdamW-RMSNorm)`

与同样 30B 调度的 Muon 两组对照：

`Delta_Muon = loss(Muon-SiLU) - loss(Muon-RMSNorm)`

如果两种优化器下 RMSNorm 均有稳定收益，支持它的作用不限于 Muon；若差距变化，提示激活与优化设置存在交互，但单 seed 不足以作强因果结论。这里 Muon LR=5e-3、AdamW LR=1e-3，因此比较的是两套优化配方，不是严格只改变算法且 LR 相同的消融，也不证明任一优化器已达到最优调参结果。

旧 7.5B / warmup 500 的 Precision_Ablations 不能代替这里 30B / warmup 2000 的匹配对照。现有常规 LM 日志提供 CE/PPL，BPB 需要准确的被评分文本 byte 数，不能用 CE/ln(2) 代替。

本实验是对默认优化器的明确覆盖；专用 skill 的全局默认仍为 Muon LR=5e-3。
