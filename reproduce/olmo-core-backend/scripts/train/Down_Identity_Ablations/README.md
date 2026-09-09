# Down 激活四组消融：30B / FP8

目的：区分「移除 Down 中间 SiLU」与「加入内部 RMSNorm」各自的收益。
四组只改变 `down_nonlinearity`，保留 OLMo3 原有 FFN 输出 RMSNorm。

| 组 | Down 分支 | 启动脚本 |
| --- | --- | --- |
| S | `down_nonlinear_v(silu(down_nonlinear_u(h)))` | `run_down_silu_fp8.sh` |
| I | `down_nonlinear_v(down_nonlinear_u(h))` | `run_down_identity_fp8.sh` |
| R | `down_nonlinear_v(rms_norm(down_nonlinear_u(h)))` | `run_down_rms_fp8.sh` |
| SR | `down_nonlinear_v(rms_norm(silu(down_nonlinear_u(h))))` | `run_down_silu_rms_norm_fp8.sh` |

Identity 只是返回激活输入，**不切换到 down_linear 分支**。四组的矩阵名字、形状、初始化顺序一致，内部 RMSNorm 无可学习 gain。网络中原有的其他 RMSNorm 不受此开关影响。

## 固定设置

| 参数 | 值 |
| --- | --- |
| global batch | 256 条序列 = 2,097,152 tokens/optimizer step |
| sequence length | 8192 |
| token budget | 30,000,000,000（约 14,306 optimizer steps） |
| optimizer / LR | 原有 Muon 配置 / 0.005 |
| WSD | 2000 step warmup，最后 20% decay；沿用现有 WSD 类的线性 decay |
| precision | Float8 enabled，其余精度策略沿用 baseline；不等于所有算子都是 FP8 |
| checkpoint | 每 1788 steps（约 12.5%）；不删除历史进度 ckpt |
| rank microbatch | 32768 tokens = 4 条序列；通过累积保持 global batch |
| Q/K/V/O rank | 各 512，不共享 |
| Gate/Up | rank 512，不共享，中间激活 SiLU |
| Down | rank 512，中间激活按 S/I/R/SR 选择 |
| 额外线性分支 | `r1=0, down_r1=0` |
| 控制函数 | `standard / none`，不加额外 clipping |

其余设置复用 `_pretrain_common.py`：数据集及顺序 seed 34521、weight decay 0.033、betas (0.9, 0.95)、gradient clipping 1.0、z-loss 1e-5、compile、HSDP/BF16 参数/FP32 梯度归约、LM evaluation 每 500 steps。模型初始化使用相同的原有默认 seed，启动时应保存并核对 resolved config，不要对某组单独覆盖。

## 运行与检查

四组脚本与之前一样使用 prepare、hceph 和分布式环境变量。可用现有 `gpu_run` 提交包装器，分别提交上述四个脚本；脚本本身不负责申请 GPU。

默认 run_name 为 `olmo3-1b-down-activation-{silu,identity,rms,silu-rms-norm}-fp8-30b`，与之前 Precision_Ablations 区分。重复提交时显式设置新的 `RUN_NAME`，避免复用保存目录；本次没有自动提交任何任务。

开跑前核对实际导入的是本仓库更新后的 olmo_mose（含 identity 和 silu_rms_norm）；模型日志中 Down 两个投影应仍为 Float8Linear，维度 8192→512→2048。已有 S/R 如果是 7.5B horizon / warmup 500，**不能直接当本次 30B / warmup 2000 的匹配对照**，应重跑 S/R。

## checkpoint 与早期观察

每 1788 steps 保存一次，周期点为 1788、3576、5364、7152、8940、10728、12516、14304。
这是按整步近似的 12.5% 间隔；框架还可能保存初始和最终 checkpoint，不能把它理解成严格只生成八个文件。

warmup 在 step 2000（4.194304B tokens）结束，因此首个 ckpt 约 3.75B 仍在 warmup。
可以在第二个 ckpt step 3576（7.499415552B tokens）做早期比较，但 **不要把 max_duration 改成 7.5B**，否则 WSD 的衰减时间也可能提前。若观察到 7.5B 就暂停，应保留 30B 的调度终点和恢复状态。

## 分析计划

在完全相同的 step/tokens、评测数据和评测精度下比较 train CE、validation CE/PPL、各验证集结果。已有常规 LM 验证输出不是 BPB；BPB 需要准确对应被评分文本的 byte 数，不能直接用 CE/ln(2) 替代。下游带 bpb 名称的任务也不能混称为整个验证集的 BPB。

用相同指标定义 `G_remove = L_S - L_I`，`G_norm = L_I - L_R`：

| 现象 | 解释 |
| --- | --- |
| I ≈ R < S | 移除 SiLU 可以解释主要收益 |
| R < I < S | 移除 SiLU 和内部归一化都有收益 |
| R < I ≈ S | 单纯移除 SiLU 不够 |
| 差异小且不稳定 | 当前证据不足，补配对 seeds / 更长观察 |

建议汇总各 12.5% 阶段附近的共同评测点、最后 10% train CE 均值、最后一个共同验证点、逐验证集差值；同时记录吞吐、显存峰值、grad norm、异常 loss。单 seed 的时间波动不能充当独立重复实验的置信区间。

单分支、无 bias、无内部 gain、忽略 epsilon 的精确算术下，外部 RMSNorm 会抵消内部归一化带来的正尺度。FP8 量化、epsilon、舍入及训练轨迹可能打破这一等价；`G_norm` 不能直接全部归因于 FP8 硬件累加误差。

## 新增 SR：SiLU → RMSNorm

令 `a = down_nonlinear_u(h)`，SR 严格执行
`z = RMSNorm(SiLU(a))`，再交给 `down_nonlinear_v`。
不是 `SiLU(RMSNorm(a))`，也不是两条分支相加。内部 RMSNorm 仍无可学习 gain，eps=1e-5。Down 两矩阵仍为 8192→512→2048，其他激活、seed、参数名与初始化不变。

四组构成 2×2 因子对照：

| | 不加内部 RMSNorm | 加内部 RMSNorm |
| --- | --- | --- |
| 不加 SiLU | I：a | R：N(a) |
| 加 SiLU | S：SiLU(a) | SR：N(SiLU(a)) |

额外统计 `L_S - L_SR`（保留 SiLU 时归一化的收益）、`L_SR - L_R`（都有归一化时 SiLU 的影响），以及
`(L_S - L_SR) - (L_I - L_R)`（内部归一化与 SiLU 的交互）。

SiLU 改变各维分量的相对关系；其后的 RMSNorm 调整整体尺度，不能把表示恢复到 SiLU 前。
同样，在无 bias、单分支、忽略 epsilon 的精确算术下，外部 RMSNorm 会抵消 SR 相对 S 的正尺度，所以除了 I/R 理想等价，S/SR 也理想等价。
实际 FP8 量化和舍入、epsilon、训练过程可能破坏等价；观察到差值不能单独证明是 Tensor Core 累加误差。
SR 与 R 保留相同的内部归一化操作，可进一步检查 SiLU 非线性本身的影响。

新增实验属于本目录的 **Muon LR=5e-3** 对照，不是 AdamW；未新增或修改 AdamW 组。
