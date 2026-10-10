# 独立UG的FFN残差与缩宽实验

2026-10-09新增。本组全程不共享参数：Attention Q/K/V/O各自rank512，低秩中间变换均为Linear；UG输入因子独立且中间Linear；保留原生SwiGLU的门控SiLU；Down低秩中间为无gain RMSNorm。用户要求减少变量，暂不测试UG共享版。

## 1. 五条配置

| 配置 | 维度主路径 | 新增宽残差 | Up→Down latent残差 | 参数量 | 作用 |
| --- | --- | --- | --- | ---: | --- |
| L1 | 2048→512→8192→512→2048 | 无，保留原block输入→输出残差 | 开 | 797,050,880 | 原宽度下测试latent短路 |
| R0 | 2048→1280→2048→1280→2048 | 关 | 关 | 797,050,880 | 同预算缩宽对照 |
| R1 | 同R0 | 开 | 关 | 797,050,880 | 宽残差贡献 |
| R2 | 同R0 | 关 | 开 | 797,050,880 | latent残差贡献 |
| R3 | 同R0 | 开 | 开 | 797,050,880 | 两类残差的组合 |

路径图只画Up主支；Gate拥有另一套独立2048→r→m因子，不能把第一个r理解为UG共享latent。L1的512短路、R2/R3的1280短路均使用Up输入因子产生的latent，不加入Gate latent，不引入新投影、可学习门或缩放参数。

R0–R3保持所有矩阵形状、参数名称、初始化顺序、model seed及数据seed一致。配置中的两个boolean不新增参数或buffer。实际GPU初始化SHA256由所有rank参与的现有callback记录；训练启动前仅声明设计相同，不提前宣称GPU逐值相等。

## 2. 精确forward及归一化顺序

采用行向量。令宽残差开关w、latent残差开关b取0或1：

```text
z_gate = x @ A_gate
z_up   = x @ A_up
g      = z_gate @ B_gate
u      = z_up   @ B_up
h      = SiLU(g) * u
s      = h + w*x                  # w=1时必须m=d=2048
t      = s @ A_down + b*z_up
f      = RMS_no_gain(t) @ B_down + w*h
y      = x + OLMo_FFN_output_norm(f)
```

OLMo原来的Attention输出Norm、FFN输出Norm、QK Norm、RoPE及残差顺序全部保留。Down RMS作用于latent相加之后，eps=1e-5；无gain、无bias。OLMo原block的FFN输出Norm仍在整个更新f之后，保留原参数及eps。

此前讨论图省略了块级Norm。实际实现显式采用上面的顺序：w=1同时开启输入2048→中间2048与中间2048→FFN输出的两条短路；最后的输入→输出连接由原block提供。**不会把已经包含输入x的完整状态作为FFN更新再次返回，避免外层重复加x。** 有块级Norm时不能把实际公式简写成未经归一化的y=s+Down(s)，也不能声称是两个分别归一化的独立FFN子层。

L1没有任何8192↔2048的宽跳连，只增加512→512短路。所有开关系数固定为1，无学习参数、额外loss或新RMS gain。

## 3. 同参数预算

d=2048。UG独立时，三个投影Up、Gate、Down的每层参数为：

```text
P_FFN = 3*(d+m)*r
原宽度：3*(2048+8192)*512  = 15,728,640
新宽度：3*(2048+2048)*1280 = 15,728,640
Attention：4*(2048*512+512*2048) = 8,388,608
每block投影：24,117,248
```

层数16、embedding、LM head及其他参数不变，五条全模型均严格797,050,880。新增残差为加法，不改变参数量。不能套用共享UG的r1536预算，也不是与1.485B原生Dense做同参数匹配。

r1280仍小于2048，但对单个2048×2048投影，两个因子的参数比完整Dense矩阵多25%；本组压缩的是相对于原8192展开的整体FFN，不声称每个方阵因子化都更省参数。FFN投影MAC预算相同；较小高维激活与更宽latent的最终显存/速度影响需实测。

## 4. 对照和结果解释

- 既有P0是不共享、Attention中间Linear、UG中间Linear、Down RMS、各rank512的旧背景，末1000步Train CE约2.480162。可作为L1原宽度参考；其原始因子分支命名、初始化和16/32卡运行记录不同，不能宣称与L1实际step0逐值相同。
- C0为K-only SiLU，C9为Q-only SiLU；这批Attention全部Linear，不能把C0/C9当成只有残差不同的对照。
- R0→R1在相同矩阵上测试两条宽跳连的组合；R0→R2测试latent短路；R1→R3及R2→R3补齐交叉对照。
- 同口径交互差分为(CE_R3−CE_R1)−(CE_R2−CE_R0)。单种子结果仅描述效果，不宣称统计显著。
- L1→R2同时改FFN展开宽度与rank，测的是同预算容量重分配整体效果。
- 宽残差允许h绕过Down读出进入FFN更新；仅latent残差仍经过最终Down因子，不能据此声称整个FFN更新不再有输出rank上限。
- 已有W1/W2是共享UG背景的缩宽实验，CE均变差。它们提供风险线索，不能代替本组独立UG残差对照。

主要指标：完整14306步Train CE及steps13307–14306的末1000步未平滑均值。保留异常原始行，任何剔除都对比较组使用相同掩码；同时列末步原值，不用W&B采样曲线或单个batch定胜负。后续benchmark复用现有OLMES任务hash，训练CE不等于benchmark收益。

## 5. 固定训练配方

| 项目 | 值 |
| --- | --- |
| 残差维度 / 层数 | 2048 / 16 |
| Attention | Q/K/V/O独立rank512，中间全部Linear |
| UG | 两个独立输入因子，中间Linear，原生SwiGLU保留 |
| Down | rank512或1280，中间无gain RMSNorm |
| 全局batch / seq | 256 sequences / 8192，2,097,152 tokens |
| 每卡microbatch | 32768 tokens；8卡累积8次 |
| 训练预算 | 30B tokens，最终step14306 |
| 优化器 | Muon LR0.005 |
| 日程 | WSD，2000步warmup，末20% decay |
| 精度 | FP8转换启用，其他运算沿用现有混合精度 |
| 种子 | model.init_seed=0、运行init_seed=12536、data seed34521 |
| checkpoint | 每1788步、保留全部 |
| 训练资源 | arch pool，单节点8卡，priority1；不在agent训练 |
| 编译缓存 | 每节点独立/tmp目录，避免共享tmp配额 |

## 6. 入口、W&B与HF支持

| 配置 | 入口 | W&B完整名称 |
| --- | --- | --- |
| L1 | [run_l1.sh](run_l1.sh) | olmo3-1b-ffn-residual-l1-m8192-r512-unshared-linear-down-rms-fp8-30b |
| R0 | [run_r0.sh](run_r0.sh) | olmo3-1b-ffn-residual-r0-m2048-r1280-unshared-linear-down-rms-fp8-30b |
| R1 | [run_r1.sh](run_r1.sh) | olmo3-1b-ffn-residual-r1-m2048-r1280-unshared-linear-down-rms-fp8-30b |
| R2 | [run_r2.sh](run_r2.sh) | olmo3-1b-ffn-residual-r2-m2048-r1280-unshared-linear-down-rms-fp8-30b |
| R3 | [run_r3.sh](run_r3.sh) | olmo3-1b-ffn-residual-r3-m2048-r1280-unshared-linear-down-rms-fp8-30b |

配置及参数保护：[CFP](../../../cfps/BS256_Seq8K_30B_WSD_FP8/FFN_Residual_Bottlenecks/_common.py)。
原生结构：[residual_feed_forward.py](../../../src/olmo_mose/residual_feed_forward.py)。
新类已注册到统一HF exporter的[对应架构模块](../../../src/olmo_mose/hf_export/architectures/residual_feed_forward.py)，独立runtime保留两个开关及Up latent来源；未编写每实验专用converter。导出必须读取真实checkpoint保存的配置。

本地检查：

```bash
python scripts/train/FFN_Residual_Bottlenecks/validate_configs.py
python -m pytest -q tests/test_residual_feed_forward.py
```

[CONFIG_AUDIT.json](CONFIG_AUDIT.json)是meta结构、真实唯一参数量和保存配置重建检查，不执行forward或读取checkpoint。
原生微型行为回归覆盖关闭残差与现有MoSE逐值等价、零Down编码器下latent通路、零Down读出下宽通路、非法形状与共享拒绝。
HF forward/缓存decode/分片checkpoint重载的微型回归已加入tests/test_hf_export.py，沿用用户暂不验证要求，数值状态为**not_run**，不把meta审计称作HF数值验证通过。

[实际提交与监控状态](RUN_STATUS.md)；本地monitor.py每1800秒检查本组实际任务以及C9，调度与训练日志分开记录。排队或SSH超时不重复提交；停止后先核实active=0、保存并同步精确W&B段，再处理授权重启。
