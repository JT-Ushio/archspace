# 共享低秩FFN：展开宽度、两端瓶颈与Q/K非线性

文档更新：2026-10-09。五条均已完成30B，最终step14306保存成功，训练日志与调度器Succeeded/active=0均已核实。本次按用户最新要求仅分析末1000步Train CE：缩宽组合未超过C2/C6，G1为2.473732，较C6低0.001031；小幅收益待重复核实。新增checkpoint评测已停止。本文保留原设计，实际结果见[末期Train CE报告](COMPLETED_RESULTS_20261009.md)，job ID见[RUN_STATUS](RUN_STATUS.md)。

## 1. 研究目标与已有证据

项目目标是在所有投影独立rank512模型的参数预算内，选择更好的从头预训练结构，尽量接近较大参数量的Dense。当前低秩预算为797,050,880参数，Dense为1,484,916,736参数；不能把二者称为同参数量。

主要选型指标仍是**完整30B训练末1000步平均Train CE**，验证用修正有效目标mask后的独立最终评估。已经得到的参考为：

| 参考 | 核心配置 | 末1000步Train CE | 修正后最终验证macro CE |
| --- | --- | ---: | ---: |
| C2 | K SiLU；Q Linear；UG共享568；Down520 RMS | 2.475668 | 2.732681 |
| C6 | Q/K SiLU；UG共享568；Down520 RMS | **2.474763** | 2.734855 |
| P0QK | Q/K SiLU；UG不共享512；Down512 RMS | 2.477410 | 2.736912 |
| F3QK | Q/K SiLU；UG共享480；Down672 RMS | 2.477421 | 2.737175 |

C2/C6还将O rank改为496，其余Q/K/V rank512；P0QK/F3QK的O为512。Train CE在共同排除step12516后统计，末窗1000点。二者差距很小，新候选需完整统计及独立权重初始化重复后再作结论。

此前十组F实验固定FFN展开宽度8192，改变UG/Down秩。没有配置超过C6；F3QK与不共享512／512参考近乎持平，大幅削减UG、提高Down的配置更差。因此下一步不再只在固定宽度8192下做单边预算转移，而检验能否通过降低展开宽度，同时扩大输入和输出瓶颈。

参考来源：[项目overview](../../../PROJECT_OVERVIEW.md)、[9月17日之后的完整报告](../../../EXPERIMENT_SUMMARY_20260907_20261007.md)、[十组F统一统计](../FFN_Rank_Reallocation/analysis/FFN_ALL_WITH_REFERENCES_FULL_TRAIN_CE.json)、[修正验证报告](../../analysis/Corrected_LM_Evaluation/RESULTS.md)。

## 2. 现有FFN究竟计算什么

令X为经过外围归一化后的FFN输入，残差维度d=2048。共享编码与分支读出为：

\[
S=XA_s,\quad U=SB_U,\quad G=SB_G,\quad H=\mathrm{SiLU}(G)\odot U.
\]

\[
T=HA_D,\qquad F(X)=\mathrm{RMSNorm}(T)B_D.
\]

矩阵尺寸如下，采用数学上的行向量约定；PyTorch Linear权重按转置形式存储。

| 矩阵 | 数学尺寸 | 功能 |
| --- | --- | --- |
| A_s | d×r_s | Up/Gate共用的输入编码 |
| B_U、B_G | 各r_s×m | 两个独立分支的宽空间读出 |
| A_D | m×r_d | 将原生SwiGLU结果编码到Down latent |
| B_D | r_d×d | 读回残差维度 |

UG的“中间Linear”指S之后不额外加SiLU，不表示删除原生SiLU(G)⊙U。Down的RMSNorm作用于r_d维latent，没有可学习gain。共享A_s不等于共享B_U/B_G，更不等于令U和G输出相同。

C2/C6的结构为2048→共享568→两个8192分支与SwiGLU→520→2048。本次问题是：在同预算下，8192维非线性展开与568／520两个瓶颈的比例是否合理？高维展开可能有价值，不能事先断言它冗余。

## 3. 主实验的假设和参数推导

假设：当前预算可能过多用于宽空间读出和Down输入矩阵，而限制了输入信息与输出变化的latent容量。减少m所节省的预算，可以同时提高r_s和r_d。这是待验证假设，不是已有结论。

在无bias、无额外gain的共享FFN中：

\[
P_F=dr_s+2mr_s+mr_d+dr_d
=(d+2m)r_s+(m+d)r_d.
\]

保持C2的Attention、embedding、归一化、层数和残差维度不变，只改变m、r_s、r_d：

| 配置 | m | r_s | r_d | UG投影参数/层 | Down投影参数/层 | 总FFN参数/层 | 全模型参数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 已有C2/C6 | 8192 | 568 | 520 | 10,469,376 | 5,324,800 | 15,794,176 | 797,050,880 |
| **W1：候选A** | **6144** | **736** | **640** | **10,551,296** | **5,242,880** | **15,794,176** | **797,050,880** |
| W1-QK：Q激活交叉对照 | 6144 | 736 | 640 | 10,551,296 | 5,242,880 | 15,794,176 | 797,050,880 |
| W1-M：只缩宽，较小预算 | 6144 | 568 | 520 | 8,142,848 | 4,259,840 | 12,402,688 | 742,787,072 |
| W2：候选B | 4096 | 976 | 944 | 9,994,240 | 5,799,936 | 15,794,176 | 797,050,880 |

W1的预算为14,336×736+8,192×640；W2为10,240×976+6,144×944。两者都精确等于15,794,176；W1-QK与W1完全同shape、同参数量。W1-M故意不匹配预算，用来解释缩宽的作用。真实16层meta模型已经逐张参数计数核对，不只依赖公式或配置声明；结果见[local_validation.json](local_validation.json)。

W1减少25%的展开宽度，W2减少50%，同时加宽两端latent；这两个百分比只描述m，不是总显存或速度的改善比例。线性矩阵乘法的参数预算仍相同，不能由展开宽度减少直接推导训练加速。

### 3.1 可以解释的结构性质

FFN分支从输入到输出依次经过r_s和r_d维状态，链式法则给出：

\[
\operatorname{rank}\left(\frac{\partial F}{\partial X}\right)
\leq\min(r_s,r_d).
\]

基线、W1、W2的上界分别为520、640、944。它是局部Jacobian的秩上界，不是实际有效秩；残差更新X+F(X)的Jacobian包含恒等项，也不受同样上界限制。更大的上界不能保证更低CE，缩小非线性展开也可能损失质量。

### 3.2 本次实验能回答什么

若W1/W2更好，可以说同预算下“减少展开、扩大两端瓶颈”的组合有效；不能仅归因于m、r_s或r_d中的一个。若更差，说明这些组合没有改善当前设置，仍不能证明8192是全局最优宽度。

本轮已经增加两条解释性对照W1-QK和W1-M，优先级高于W2/G1。它们的用途如下；围绕最优点的更多搜索仍留待首轮质量结果。

### 3.3 W1-QK：形成完整的Q激活交叉对照

W1-QK只将Q的低秩中间变换从Identity改为SiLU，K仍为SiLU。Attention rank、FFN形状、初始化规则、优化器、batch、数据seed和训练日程都与W1相同。它不共享Q/K输入因子，不引入任何参数。

| FFN配置 | Q Linear、K SiLU | Q/K都SiLU |
| --- | --- | --- |
| m8192／共享UG568／Down520 | 已有C2 | 已有C6 |
| m6144／共享UG736／Down640 | W1 | W1-QK |

四条均为797,050,880参数。交叉比较可以区分同预算宽度方案在两种Q激活背景下的作用，而不是只拿C2背景的W1与C6比较。

对同一指标、同一训练窗口、同一异常掩码计算：

\[
\Delta_{\mathrm{interaction}}
=(L_{\mathrm{W1\text{-}QK}}-L_{\mathrm{W1}})
 -(L_{\mathrm{C6}}-L_{\mathrm{C2}}).
\]

这里负值表示Q SiLU在W1形状下带来的CE下降更大，正值表示下降更小或更不利。它是差分描述；需要独立权重种子重复、报告差异的波动，单个种子不能宣称交互显著。主指标是末1000步Train CE，完整训练均值和最终修正验证各自另算交互，不混在同一个差分中。

**W1与W1-QK的初始化检查**：相同shape、相同模块参数注册顺序，Identity/SiLU均无参数且不消耗初始化随机数。不能仅凭这句话宣称逐值一致；[validate_initialization.py](validate_initialization.py)实际调用原生完整16层初始化器，在CPU上对所有参数和注册buffer逐张执行torch.equal，另检查有限值，结果保存到[initialization_validation.json](initialization_validation.json)。这是FP32、FP8/FSDP转换前的本地验证。正式GPU任务在首次step0的pre_train回调、dry-run前，所有rank共同还原每张参数；仅rank0对全部原始字节记录SHA256及逐张清单，不修改权重或RNG。日志含INITIALIZATION_FINGERPRINT，清单位于run/stage下initialization_fingerprint.json；比较两条实际指纹，若不匹配须先定位原因，不能宣称初始化相同。恢复step>0时跳过，不覆盖初始记录。正式运行尽量统一W1/W1-QK的设备数、型号和代码版本，并核实没有意外加载已有checkpoint。

### 3.4 W1-M：只缩宽，不增加瓶颈rank

W1-M沿用C2的Q Linear、K SiLU、共享UG568、Down520 RMS，仅将m从8192改为6144。每层FFN投影参数为：

\[
P_{\mathrm{W1-M}}=14,336\times568+8,192\times520
=12,402,688.
\]

全模型797,050,880−16×(15,794,176−12,402,688)=**742,787,072**，比C2少54,263,808参数。它是较小模型的解释对照，不参与“严格同预算改进”的结论。

| 比较 | 改变 | 能回答的问题 |
| --- | --- | --- |
| C2 → W1-M | 只缩展开宽度，UG/Down rank不变 | 缩宽本身的质量代价和效率变化 |
| W1-M → W1 | 相同宽度6144，UG568→736、Down520→640 | 把参数回补到两端瓶颈是否有效 |
| C2 → W1 | 缩宽并回补rank，全模型预算相同 | 最终同预算方案是否改善 |

| 结果 | 可以支持的解释 |
| --- | --- |
| W1-M接近C2 | 缩宽质量代价较小，同时得到一个较小模型候选 |
| W1-M变差、W1优于C2 | 两端容量回补弥补了缩宽损失并产生净收益 |
| W1-M、W1都改善 | 缩宽本身可能有正向作用，优化/正则化机制仍待验证 |
| W1不如W1-M | 当前配方下增加瓶颈rank没有转化为质量收益 |

W1-M和W1同时改变两端rank，不能单独归因于UG或Down。C2与W1-M的矩阵shape不同，也不能要求所有未改模块的初始化逐值相同；与W1/W1-QK的同shape对照区分记录。

## 4. 完整配置：什么改变，什么固定

| 部件 | C2参考 | W1 | W1-QK | W1-M | W2 |
| --- | --- | --- | --- | --- | --- |
| Q | 独立512／Linear | 同C2 | 独立512／SiLU | 同C2 | 同C2 |
| K | 独立512／SiLU | 同C2 | 同C2 | 同C2 | 同C2 |
| V | 独立512／Linear | 同C2 | 同C2 | 同C2 | 同C2 |
| O | 独立496／Linear | 同C2 | 同C2 | 同C2 | 同C2 |
| UG输入因子 | 共享568 | 共享736 | 共享736 | 共享568 | 共享976 |
| UG中间变换 | Linear | Linear | Linear | Linear | Linear |
| FFN展开宽度 | 8192 | 6144 | 6144 | 6144 | 4096 |
| 原生门控 | SiLU(G)⊙U | 相同 | 相同 | 相同 | 相同 |
| Down latent | 520／无gain RMSNorm | 640／无gain RMSNorm | 640／无gain RMSNorm | 520／无gain RMSNorm | 944／无gain RMSNorm |
| 残差、外围归一化、RoPE | 现有OLMo配置 | 相同 | 相同 | 相同 | 相同 |

初始化沿用现有MoSE规则：输入编码矩阵使用原有normal std，读出矩阵按对应rank缩放。没有单独重调LR、增加辅助损失、修改位置编码或改变归一化gain。

相同init seed只保证固定架构的可复现性。不同矩阵shape会改变随机数消耗，不能宣称所有未改矩阵在不同模型中逐值相同。全局batch一致也不保证不同设备数/归约顺序下数值完全一致；实际硬件、分片方式和恢复历史需要保存。

## 5. 独立备选：G1的可学习Q/K非线性

G1不修改宽度或FFN秩，保留C2的8192／568／520，只增加：

\[
Q=\phi_{\alpha_{Q,\ell}}(XA_{Q,\ell})B_{Q,\ell},\qquad
K=\phi_{\alpha_{K,\ell}}(XA_{K,\ell})B_{K,\ell},
\]

\[
\phi_\alpha(a)=a+\alpha(\mathrm{SiLU}(a)-a).
\]

每层两个标量，16层增加32参数；alpha_Q初始化为0、alpha_K为1，初始函数对应C2。Q与K输入矩阵独立，不做QK共享。V/O、UG和Down保持C2配置。

假设：不同层可能需要不同强度的非线性，固定Q Linear／K SiLU只是一个统一选择。已有Q激活收益较小且训练/验证方向相反，因此G1优先级低于宽度实验。

### 5.1 实现约定

- 模型参数量为797,050,912，比C2多32；明确列为近似同预算，不伪称严格匹配。
- 每个标量存成shape(1)的Parameter，支持FSDP分片；它进入现有Muon配置的AdamW、无weight decay组，沿用训练LR与WSD日程。
- 不做sigmoid、截断或额外正则。alpha可以小于0或大于1，这时是外推而非Identity与SiLU之间的凸组合。
- 实现采用等价的(1-alpha)*a+alpha*SiLU(a)。SiLU在原激活dtype计算，混合用FP32并转回，避免端点减法带来的数值差别。
- 从meta模型materialize后，初始化器会明确重设alpha。它和对应optimizer状态随checkpoint保存；不会自动加载C2旧run作为G1的训练输出。
- 每100步记录每层Q/K alpha及各自mean/min/max。DTensor标量由全部rank进入还原操作后记录，包含本地分片为空的rank。

CPU检查已验证初始输出和输入梯度与C2相同、alpha梯度有限并可更新、checkpoint恢复一致、图捕获无断点，以及双进程空分片日志正确。尚未执行GPU上的分布式FP8训练，实际启动后仍须核实这一流程。

### 5.2 怎样解读alpha

alpha接近0表示该投影倾向Identity，接近1表示倾向SiLU；不同层分布可以作为描述性证据。如果alpha变化但CE不改善，不把变化本身当作架构收益。如果alpha几乎不动，也不能单凭这一点证明固定端点最优，仍需检查梯度、更新尺度和优化状态。

是否保留G1由质量决定，不由alpha曲线是否“有趣”决定。不要在本轮直接把G1和W1/W2合并，否则无法判断收益来源。

## 6. 五条实验统一训练配方

| 设置 | 数值 |
| --- | --- |
| 层数／残差维度 | 16／2048 |
| 序列长度 | 8192 tokens |
| Global batch | 256序列／2,097,152 tokens |
| 每卡microbatch | 32768 tokens／4序列 |
| 日程 | 30B tokens，约14306 optimizer steps |
| 优化器 | 原有Muon配置，LR0.005 |
| WSD | warmup2000步，末20% decay |
| 精度 | FP8及原有mixed precision |
| 运行级随机seed／数据seed | 12536／34521 |
| 模型权重初始化seed | model.init_seed=0，沿用现有C2配置 |
| Checkpoint | 每1788步，保留全部，另含step0与最终保存 |
| 已验证global batch整除 | 8／16／32卡，分别累积8／4／2次 |

运行级config.init_seed与模型的model.init_seed是两个字段：前者用于seed_all，原生权重初始化器另用model.init_seed创建Generator。本次保持与既有C2配置一致，分别记录12536和0，不将更改运行级seed等同于独立权重初始化。未来做独立权重seed重复时，应显式更改model.init_seed并对参考配置同样核实。

不通过减少microbatch改变global batch，不通过缩短max_duration把30B schedule改成短日程。中途观察以同step/tokens比较；早期判断只能用于发现失败，不能替代完整终点质量结论。

## 7. 运行顺序与对照条件

用户已恢复全部五条提交，按下列顺序执行：

1. **W1**：首先检验较温和的宽度与瓶颈调整。
2. **W1-QK**：优先与W1组成同shape、同初始化的Q激活对照，尽量使用相同硬件配置。
3. **W1-M**：补全“缩宽→容量回补”的解释链，明确为较小模型。
4. **W2**：资源允许，再测试更强的同预算宽度调整。
5. **G1**：作为独立门控对照，本轮也已获提交授权。
6. **重复**：首轮完成后，仅给有希望的候选增加真正独立的model.init_seed；对应C2/C6参考也核实同一权重seed字段，data seed固定34521。

首轮使用既有C2/C6参考，不新增Dense baseline。若运行环境、数据或恢复方式出现无法对齐的差异，再将其作为需要补对照的具体原因记录，不能用相同run名覆盖旧结果。

资源默认arch 2×8低优；按用户既有偏好，arch需排队就放弃该次候选提交，可用pool 1×8慢跑。任何加速都不改变配置和global batch。实际job名、提交响应、设备数、分片方式、W&B段和checkpoint路径必须写入台账，不能把排队当作停止而重复提交。

## 8. 启动与每30分钟监控

下面是任务启动后的核实要求。五条均已获得实际job ID；本地monitor.py --loop已启动，每1800秒查询一次真实状态和日志，并重新预测arch16卡/pool8卡资源。实际运行证据见RUN_STATUS.md、latest_status.json及monitor_history。

| 阶段 | 核实内容 | 证据 |
| --- | --- | --- |
| 调度 | 任务真实状态、active replicas、8/16卡配置 | 调度器状态及实际job ID |
| 初始化 | 参数审计、从头初始化、step0保存、dry-run完成 | 已保存配置及训练日志 |
| 分布式 | 16卡对应16 ranks、dp_replicate2/dp_shard8；8卡为1/8 | 训练mesh日志 |
| 首步 | CE、梯度范数有限；没有训练OOM/Traceback/NaN | 第一段实际训练日志 |
| W1/W1-QK对照 | 实际step0权重逐值一致，硬件与分片配置一致 | 两份step0 checkpoint的后续核对结果 |
| G1特有 | Q/K alpha初值0/1，step100起出现逐层指标 | qk_alpha曲线及checkpoint参数 |
| 周期检查 | optimizer step、累计tokens、CE、吞吐、磁盘写入、状态 | 每30分钟一次观察台账 |
| 完成 | step14306最终保存成功且Training complete；active=0 | 终点日志与调度器终态 |

任务停止时，先核实终态与active=0，再解析该attempt的精确离线W&B段，调用wandb_sjy同步并核实完成；只有在已授权恢复的范围内才续训。不得因为SSH或日志查询超时重复提交。继续沿用原run与stage恢复，不换短日程，也不静默改变设备数、seed或模型维度。

## 9. 质量分析口径与通过条件

### 9.1 主要Train CE

对每条完成run读取全部steps1–14306的Train CE，合并已核实续训段并检查覆盖、重复、非有限/非正值。保留原始记录；跨run比较使用统一异常掩码，另报告原始敏感性结果。末1000步固定为13307–14306，不用W&B平滑值，也不用最后一个batch代替。

对C2的差值用于判断C2背景内是否改善；对C6的差值用于判断是否产生新的低秩主配置。W1对C2、W1-QK对C6分别是同背景对照。W1-M因预算不同，单列为较小模型，不能与严格同预算候选混为一类。若只低于C2、仍高于C6，应写为背景内改善。差异很小时，先补种子重复，不提前声称稳定收益。

### 9.2 最终验证与下游

最终checkpoint在相同11源、相同有效目标mask下独立重评，与C2/C6修正后结果比较。旧训练内验证mask的绝对值不能直接混入。训练更低而验证更高时，分别报告，不把Train CE选型包装成全面质量胜出。下游评估需固定题集、提示、batch和评估代码后再作为论文证据。

### 9.3 决策表

| 观察到的结果 | 下一步与可写结论 |
| --- | --- |
| 同预算候选W1/W1-QK/W2低于C6，验证也改善，独立权重种子支持 | 优先作为新主候选，补统一效率与下游评估 |
| 仅优于C2、未超过C6 | 保留为C2背景内改善，主配置仍为C6 |
| Train CE改善但验证变差 | 报告训练/验证取舍；不宣称全面更优 |
| CE接近、显存或吞吐有重复改善 | 可以作为效率候选，但须给出实测条件与质量差值 |
| W1/W1-QK/W2均明显更差 | 当前已测宽度/瓶颈组合不利，停止追加大量邻近点 |
| W1-M质量接近、显存或吞吐更好 | 单列较小模型的质量/效率取舍，不称同参数改进 |
| G1的alpha变化但质量未改善 | 作为负结果或分析线索，不升级为主贡献 |

## 10. 效率测量，避免把参数匹配当效率匹配

宽度减少可以降低部分高维FFN激活；更宽latent、FP32门控混合、矩阵形状和编译kernel又可能增加其他开销。本次没有预设显存节省或加速比例。

先在同一设备与节点数、相同microbatch、精度和测量流程下做训练步微基准。统一预热并同步设备计时，报告tokens/s、平均步时、峰值allocated与reserved，记录是否包含optimizer更新。不能将8卡和16卡的裸吞吐直接作为架构加速比。

再从真实训练窗口统计包含数据加载、eval和checkpoint的实际进度。微基准与实际训练效率分别报告，并保留GPU型号、软件版本、编译缓存、checkpoint/数据路径等测量条件。质量筛选没有正结果时，仍可以记录效率，但不以参数减少替代显存测量。

## 11. 文件、计划W&B名称与结果台账

| 实验 | 配置入口 | Launcher | 计划W&B完整名称 |
| --- | --- | --- | --- |
| W1 | [w1.py](../../../cfps/BS256_Seq8K_30B_WSD_FP8/FFN_Width_Ablations/w1.py) | [run_w1.sh](run_w1.sh) | olmo3-1b-ffn-width-w1-m6144-ug736-down640-c2-k-silu-down-rms-fp8-30b |
| W1-QK | [w1qk.py](../../../cfps/BS256_Seq8K_30B_WSD_FP8/FFN_Width_Ablations/w1qk.py) | [run_w1qk.sh](run_w1qk.sh) | olmo3-1b-ffn-width-w1qk-m6144-ug736-down640-c6-qk-silu-down-rms-fp8-30b |
| W1-M | [w1m.py](../../../cfps/BS256_Seq8K_30B_WSD_FP8/FFN_Width_Ablations/w1m.py) | [run_w1m.sh](run_w1m.sh) | olmo3-1b-ffn-width-w1m-m6144-ug568-down520-c2-k-silu-down-rms-fp8-30b |
| W2 | [w2.py](../../../cfps/BS256_Seq8K_30B_WSD_FP8/FFN_Width_Ablations/w2.py) | [run_w2.sh](run_w2.sh) | olmo3-1b-ffn-width-w2-m4096-ug976-down944-c2-k-silu-down-rms-fp8-30b |
| G1 | [g1.py](../../../cfps/BS256_Seq8K_30B_WSD_FP8/FFN_Width_Ablations/g1.py) | [run_g1.sh](run_g1.sh) | olmo3-1b-qk-adaptive-g1-c2-q0-k1-m8192-ug568-down520-down-rms-fp8-30b |

这些是配置中的run名称；五条GPU任务已提交，在线W&B创建与同步情况仍以实际日志为准。每条checkpoint保存到独立的run/stage目录，具体提交命令与环境说明见[README](README.md)，预设资源和状态见[submission_plan.json](submission_plan.json)。

质量结果台账模板：

| 配置 | init/data seed | 实际设备数 | 完成步数 | 全程Train CE | 末1000步Train CE | 最终修正验证 | W&B/完整历史来源 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| W1 | 12536/34521 | 待调度 | 待训练 | 待统计 | 待统计 | 待评估 | 待创建 |
| W1-QK | 12536/34521 | 待调度 | 待训练 | 待统计 | 待统计 | 待评估 | 待创建 |
| W1-M | 12536/34521 | 待调度 | 待训练 | 待统计 | 待统计 | 待评估 | 待创建 |
| W2 | 12536/34521 | 待提交 | 待训练 | 待统计 | 待统计 | 待评估 | 待创建 |
| G1 | 12536/34521 | 待提交 | 待训练 | 待统计 | 待统计 | 待评估 | 待创建 |

效率结果台账模板：

| 配置 | 设备/节点数 | 预热/测量窗口 | 实测tokens/s | 步时 | 峰值allocated/reserved | 是否含数据/eval/保存 |
| --- | --- | --- | --- | --- | --- | --- |
| W1 | 待测 | 待测 | 待测 | 待测 | 待测 | 待说明 |
| W1-QK | 待测 | 待测 | 待测 | 待测 | 待测 | 待说明 |
| W1-M | 待测 | 待测 | 待测 | 待测 | 待测 | 待说明 |
| W2 | 待测 | 待测 | 待测 | 待测 | 待测 | 待说明 |
| G1 | 待测 | 待测 | 待测 | 待测 | 待测 | 待说明 |

## 12. 当前验证与论文表述

本次扩展的9项CPU/meta检查通过：五条真实16层模型参数量和C2训练配方、8/16/32卡global batch整除、W1/W1-M/W2实际FFN前向和反向，以及W1-QK只改变Q激活的配置检查。另用两个完整模型的原生CPU FP32初始化，对275张参数、共797,050,880个元素执行torch.equal，全部逐值一致，有限值检查通过；没有注册buffer。结果见[本地核对结果](local_validation.json)与[初始化比较](initialization_validation.json)。此前G1相关的初始化函数/输入梯度、alpha更新、checkpoint恢复、无weight decay分组、图捕获及双进程空分片日志已随原11项检查通过，本次未重复执行未改动的G1训练实现。六个shell脚本通过bash -n，配置、审计和测试文件通过ruff，Markdown表格和相对链接已核对。2026-10-08 08:58已核实五条GPU分布式FP8训练启动：pool8卡H200、8 ranks、从头初始化、step0保存成功、dry-run完成，CE与梯度有限；W1/W1-QK起始275张参数、797,050,880个元素的完整字节SHA256一致。见[GPU启动证据](gpu_startup_validation.json)。尚未产生新30B终点质量或效率结论。

如果主实验成立，论文可以讨论共享低秩SwiGLU中展开宽度与两端瓶颈的同预算分配，以及得到的可重复质量/效率结果。不要把单独调整宽度、共享矩阵或一个可学习标量门控作为新发明。若结果不成立，则将其作为明确边界：已测的宽度与瓶颈组合没有改善当前配置，避免扩大成所有宽度调整都无效。
