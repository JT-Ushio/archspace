# Dense 预算下的 FFN Linear + Down RMS 消融

本组以原生 Dense OLMo3-1B 为参照，分别尝试 Up、Gate、Up+Gate 的线性因子化，同时将 Down 改为带 RMSNorm 的因子化。Attention 的 Q/K/V/O 全部保持 Dense，无输入因子共享；它不是全投影 rank512 的 C0–C8 系列。

## 配置与参数预算

残差宽度 d=2048，FFN 展开宽度 m=8192，16 层。被替换投影的 rank 均为 1632，Down RMSNorm 无 gain、无 bias，eps=1e-5。未被替换的投影保留原 Dense 结构。

| 标签 | Up | Gate 的输入投影 | Down | 全模型参数 | 比 Dense 少 |
| --- | --- | --- | --- | ---: | ---: |
| DU | Linear 因子，r1632 | Dense | RMS 因子，r1632 | 1,482,819,584 | 0.1412% |
| DG | Dense | Linear 因子，r1632 | RMS 因子，r1632 | 1,482,819,584 | 0.1412% |
| DUG | Linear 因子，r1632 | 独立 Linear 因子，r1632 | RMS 因子，r1632 | 1,481,771,008 | 0.2118% |
| 已有 Dense 对照 | Dense | Dense | Dense | 1,484,916,736 | 0 |

单个 Dense 投影有 dm=16,777,216 权重；因子化有 (d+m)r=16,711,680 权重。严格同参数需要 r=dm/(d+m)=1638.4，不存在整数 rank 解。这里沿用此前 Dense Up/Gate 消融接受的 rank1632 向下取整，**明确命名为 near-matched，不称为严格等参数**。不额外添加参数填充预算，也不改 embedding、FFN 宽度或 Attention。

三组之间的 rank 相同，但 DUG 的全模型参数略少于 DU/DG；对比必须保留这一差异。也没有将 UG 绑定成共享编码器。

2026-10-09重新核对：在本机分别实例化Dense、DU、DG、DUG完整原生meta模型，逐个参数张量统计，与配置num_params及任务台账全部一致。结果仍为近似匹配：DU/DG少2,097,152参数，DUG少3,145,728参数。没有分配GPU、加载checkpoint或运行forward评测，也没有修改正在训练的配置。[实际参数计数](PARAMETER_AUDIT_20261009.json)。

## 完整 FFN 公式

以下使用行向量；A 为输入因子、B 为输出因子，W 为原生 Dense 权重。

DU：

\[
u=xA_uB_u,\qquad g=xW_g,\qquad h=\operatorname{SiLU}(g)\odot u,
\]
\[
F(x)=\operatorname{RMSNorm}(hA_d)B_d.
\]

DG：

\[
u=xW_u,\qquad g=xA_gB_g,\qquad h=\operatorname{SiLU}(g)\odot u,
\]
\[
F(x)=\operatorname{RMSNorm}(hA_d)B_d.
\]

DUG：

\[
u=xA_uB_u,\qquad g=xA_gB_g,\qquad h=\operatorname{SiLU}(g)\odot u,
\]
\[
F(x)=\operatorname{RMSNorm}(hA_d)B_d,\qquad A_u\text{ 与 }A_g\text{ 独立}.
\]

**Gate Linear 指因子之间不加 SiLU；原生 SwiGLU 的外层 SiLU(g) 仍保留。** Down RMSNorm 只沿每个 token 的 Down latent 通道归一化，并保留已有 block 输出 RMSNorm：

\[
\operatorname{RMSNorm}(a)=\frac{a}{\sqrt{\frac1{1632}\sum_{j=1}^{1632}a_j^2+10^{-5}}}.
\]

因此本组不是删除所有 Gate 非线性，也不是只在原 Dense Down 输出上添加 RMSNorm。

## 训练与比较

采用已有 Dense 单点实验配方：30B tokens，seq8192，全局 batch256 序列 /2,097,152 tokens；Muon LR0.005，weight decay0.033，WSD warmup2000，末20% decay，FP8 开启。init seed12536，data seed34521。rank microbatch32768 tokens，在1×8卡上每步累积8次，改变卡数不改变全局 batch 或训练日程。约14306个 optimizer steps；每1788步存 checkpoint，保留全部。

新增因子使用已有 MoSE 初始化：A 的 std0.02，B 的 std1/sqrt(rank)；未改的 Dense 投影保留原初始化。与原 Dense 的 initializer 和 microbatch 差异需保留在结论里。三组改变的矩阵集合不同，不宣称 step0 全权重逐值相同。

主要指标为完整 Train CE 曲线及统一末1000步平均 CE，与已有 Dense baseline 在相同 optimizer step/token 数上比较。保留异常原始数据，使用同一清理集合做敏感性统计；验证 CE 为补充指标。三组都同时改变 Down，因此 DU/DG 对 Dense 的差异不能全部归因于 Up/Gate；若后续要拆解因果，需要另有 Down-only 对照。当前只提交用户指定的三组。

## 脚本、W&B 名称与调度

| 标签 | Launcher | 完整 W&B run 名称 |
| --- | --- | --- |
| DU | [run_up_linear_down_rms.sh](run_up_linear_down_rms.sh) | `olmo3-1b-dense-nearmatched-up-linear-r1632-down-rms-r1632-fp8-30b` |
| DG | [run_gate_linear_down_rms.sh](run_gate_linear_down_rms.sh) | `olmo3-1b-dense-nearmatched-gate-linear-r1632-down-rms-r1632-fp8-30b` |
| DUG | [run_up_gate_linear_down_rms.sh](run_up_gate_linear_down_rms.sh) | `olmo3-1b-dense-nearmatched-up-gate-linear-r1632-down-rms-r1632-fp8-30b` |

用户最新要求：**只在 arch pool（ailab-llmarchitecture / llmarchitecture_gpu_pool）排队，每个任务1×8卡，priority1，任务名前缀 xtuner-；禁止在 agent 启动训练。** 实际任务记录保存于 `jobs.json`，状态见 `RUN_STATUS.md`。

模型实现位于 `src/olmo_mose/dense_ffn_factorization.py`；配置位于 `cfps/BS256_Seq8K_30B_WSD_FP8/Dense_FFN_Linear_RMS/`。编译缓存使用节点本地临时目录，避免已出现的共享 tmp 配额问题。新 run 名称各自对应独立 checkpoint 路径，不覆盖历史实验。

## HF 转换与检查

新增精确类型 adapter：`hf_export/architectures/dense_ffn_factorization.py`。复用统一 `ffn_dense` 与低秩 projection runtime，保留独立因子、Linear 激活、Down RMSNorm 的位置/eps/无 gain 设置，不为三个实验分别建立转换器。

三个配置的序列化与 meta 模型结构、参数量和 HF state 结构检查已通过；数值 HF 验证用例也已加入 `tests/test_hf_export.py`，仍按用户先前要求暂不执行。训练公式与梯度用例位于 `tests/test_dense_ffn_factorization.py`，与 HF checkpoint 数值验证分别记录。checkpoint 出现后可使用 [统一转换入口](../../convert/README.md)。
