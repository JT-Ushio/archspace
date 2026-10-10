# C2背景下的FFN宽度与瓶颈实验

状态：2026-10-09，W1/W1-QK/W1-M/W2/G1均完成30B，step14306保存成功、Succeeded/active=0。按用户要求仅分析末1000步Train CE：G1为2.473732，略低于C6的2.474763；W1/W1-QK/W2均未超过C2/C6。W1-M为少约6.8%参数的解释对照，G1增加32个标量。新增checkpoint评测已按用户要求停止，未继续提交。完整配置、W&B名称和比较边界见[末期结果](COMPLETED_RESULTS_20261009.md)，实际监控状态见[RUN_STATUS](RUN_STATUS.md)。

详细背景、预算推导、对照条件、监控与测量口径、结果判定及台账模板见[完整实验设计](EXPERIMENT_DESIGN.md)。本页保留配置与运行入口。

## 要回答的问题

此前F组固定FFN展开宽度8192，只调整UG/Down秩。本组同时减少展开宽度、加宽输入与输出瓶颈，检验同参数量下的容量分配是否更好。它测试的是这三个维度的组合，不能将改善全部归因于缩小宽度。

以C2为激活与Attention背景：Q512/Linear、K512/SiLU、V512/Linear、O496/Linear，Attention输入因子独立；UG共享输入因子、中间Linear，保留原生SwiGLU；Down中间无gain RMSNorm，投影均无bias。层数16、残差维度2048、归一化、embedding、RoPE均沿用C2。

## 主实验、解释对照与参数预算

设FFN展开宽度为m，共享UG rank为rs，Down rank为rd，则每层FFN投影参数为：

\[
P_F=(2048+2m)r_s+(m+2048)r_d.
\]

| 配置 | 展开宽度m | UG共享rank | Down rank | FFN参数/层 | 全模型参数 | 分支Jacobian秩上界 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 已有C2/C6 | 8192 | 568 | 520 | 15,794,176 | 797,050,880 | 520 |
| **W1：候选A，优先** | **6144** | **736** | **640** | **15,794,176** | **797,050,880** | **640** |
| W1-QK：同shape的Q激活对照 | 6144 | 736 | 640 | 15,794,176 | 797,050,880 | 640 |
| W1-M：只缩宽，较小模型 | 6144 | 568 | 520 | 12,402,688 | 742,787,072 | 520 |
| W2：更强宽度调整 | 4096 | 976 | 944 | 15,794,176 | 797,050,880 | 944 |

结构为S=XA_UG，U=SB_U，G=SB_G，H=SiLU(G)⊙U，F(X)=RMSNorm(HA_D)B_D。瓶颈两端约束FFN分支的局部Jacobian：rank(dF/dX)≤min(rs,rd)。这不是实际有效秩、残差block的秩或质量提升保证。展开宽度减少可能减少高维激活，但更宽latent及kernel实现会改变开销，显存/吞吐必须实测。

W1/W2只改变FFN展开宽度与两个rank；W1-QK只比W1额外将Q改为SiLU，K保持SiLU；W1-M保持C2的UG568/Down520，仅缩宽到6144，故意减少54,263,808参数。宽度相关矩阵采用现有MoSE初始化，不增加新算子或额外损失。不同shape即使使用相同seed，也不保证未改矩阵的初始化逐值相同，因此首轮仅作筛选；若有收益，再重复初始化种子12537/12538。

## 两条解释性对照

| FFN配置 | Q Linear、K SiLU | Q/K SiLU |
| --- | --- | --- |
| 8192／共享568／Down520 | 已有C2 | 已有C6 |
| 6144／共享736／Down640 | W1 | W1-QK |

四格均为797,050,880参数。用同一指标、窗口及异常掩码计算Δinteraction=(L_W1-QK−L_W1)−(L_C6−L_C2)。单种子差分不代表交互显著。W1/W1-QK全模型初始权重使用[实际初始化审计](validate_initialization.py)逐张torch.equal核对，结果见[初始化证据](initialization_validation.json)；CPU FP32逐张torch.equal已通过；08:58另外核实实际GPU全部初始参数字节的SHA256一致，证据见[gpu_startup_validation.json](gpu_startup_validation.json)。不把该证据描述为已重新加载两份磁盘checkpoint进行逐张比较。

C2→W1-M只缩宽；W1-M→W1在相同宽度增加两端rank；C2→W1比较最终同预算方案。W1-M参数更少，结果单列：接近C2意味着缩宽代价较小；变差而W1改善意味着容量回补有效；两者都改善提示缩宽本身有益；W1反而更差说明回补未转化为收益。完整解释边界见[设计文档3.3–3.4](EXPERIMENT_DESIGN.md)。

## 备选G1：Q/K各层可学习非线性

G1保持C2的m8192／UG共享568／Down520，独立测试：

\[
\phi_\alpha(a)=a+\alpha(\mathrm{SiLU}(a)-a),\qquad
Q=\phi_{\alpha_Q}(XA_Q)B_Q,\quad
K=\phi_{\alpha_K}(XA_K)B_K.
\]

每层Q/K各一个标量，初始化alpha_Q=0、alpha_K=1；共增加32个参数，全模型为**797,050,912**，相对基线增加约0.000004%，不称严格同参数量。用一元素Parameter存储以支持FSDP分片；标量进入现有Muon配置的AdamW、weight_decay=0组。alpha不做sigmoid、截断或额外正则，允许超出[0,1]。

实现用等价的(1-alpha)*a+alpha*SiLU(a)，激活仍在原dtype计算、混合用FP32并转回；alpha端点的初始前向和输入梯度与C2一致。meta模型materialize后会显式重新设置alpha；模型checkpoint与optimizer状态均保存这些参数。原有Attention、归一化和位置处理保持不变，新模块只由G1配置使用。

每100步记录qk_alpha/layerN/q、qk_alpha/layerN/k及Q/K各自mean/min/max。分片标量由所有rank共同还原后记录，避免仅rank0调用集体操作。检查层间alpha分布及是否超出[0,1]，但是否有效仍以完整Train CE及统一验证为准。混合额外引入计算/激活开销，不因只加32个参数就声称没有运行开销。

## 配方、对照与判定

五条均设计为从头训练：seq8192，global batch256（2,097,152 tokens），每卡microbatch32768；30B／14306步，Muon LR0.005，WSD warmup2000／末20% decay，FP8；checkpoint1788步，保留全部；运行级seed12536、data seed34521；模型权重Generator使用独立字段model.init_seed=0，沿用C2。更改运行级seed不等同于权重独立初始化，未来seed重复需显式设置model.init_seed。8/16/32卡分别累积8/4/2次，global batch与日程不变。

主要指标为steps13307–14306的末1000步平均Train CE，同时保存完整14306步历史、全程均值与原始末步。比较时对齐step/tokens，保留原始异常点，并对整个比较组使用共同异常掩码，不能直接拼接不同掩码的标量。既有同口径参考为C2 2.475668、C6 2.474763；C2的修正后最终验证为2.732681，C6为2.734855。新实验需按相同有效目标mask另作最终验证，不混用旧训练内验证mask的绝对值。

当前已授权五条，按W1、W1-QK、W1-M、W2、G1顺序提交，避免同时改变宽度和门控。若仅优于C2、未超过C6，应如实写为背景内改善。统一设备、microbatch和测量流程后再比较峰值allocated/reserved、实测tokens/s；训练步微基准与含数据加载、eval、保存的实际训练效率分别记录。

## 入口和计划W&B名称

五条GPU任务已提交；以下为配置中的run名称，W&B的实际初始化和同步以日志为准。每条使用独立run/checkpoint目录，训练重启必须沿用对应名称和stage。

| 配置 | 入口脚本 | 计划W&B完整名称 |
| --- | --- | --- |
| W1 | [run_w1.sh](run_w1.sh) | olmo3-1b-ffn-width-w1-m6144-ug736-down640-c2-k-silu-down-rms-fp8-30b |
| W1-QK | [run_w1qk.sh](run_w1qk.sh) | olmo3-1b-ffn-width-w1qk-m6144-ug736-down640-c6-qk-silu-down-rms-fp8-30b |
| W1-M | [run_w1m.sh](run_w1m.sh) | olmo3-1b-ffn-width-w1m-m6144-ug568-down520-c2-k-silu-down-rms-fp8-30b |
| W2 | [run_w2.sh](run_w2.sh) | olmo3-1b-ffn-width-w2-m4096-ug976-down944-c2-k-silu-down-rms-fp8-30b |
| G1 | [run_g1.sh](run_g1.sh) | olmo3-1b-qk-adaptive-g1-c2-q0-k1-m8192-ug568-down520-down-rms-fp8-30b |

[配置](../../../cfps/BS256_Seq8K_30B_WSD_FP8/FFN_Width_Ablations/_common.py)／[真实参数审计](validate_configs.py)／[本地核对结果](local_validation.json)／[提交计划](submission_plan.json)。

## 本地核对与提交命令

在含torch/olmo_core的环境、从backend目录运行：

```bash
python scripts/train/FFN_Width_Ablations/validate_configs.py \
  --output scripts/train/FFN_Width_Ablations/local_validation.json
python -m pytest -q tests/test_width_bottleneck_ablation.py tests/test_adaptive_qk_attention.py
# 完整16层CPU FP32初始化对比，约6.4GB参数内存，不创建GPU任务
python scripts/train/FFN_Width_Ablations/validate_initialization.py \
  --output scripts/train/FFN_Width_Ablations/initialization_validation.json
```

提交时，训练优先arch 2×8、低优；用户既有策略是arch需要排队就放弃该次提交，可用pool 1×8慢跑，不能因此重复同一输出目录的活跃任务。准备launcher不代表训练已提交。实际提交时先SSH，激活成功后调用gpu_run，例如W1：

```bash
ssh -CAXY cpu-develop.jitao+root.ailab-llmagent.ws@h.pjlab.org.cn
```

在提交主机：

```bash
conda activate MoSE && PRIORITY=1 gpu_run arch 2 8 mose-ffn-width-w1-1008-unique \
  /mnt/shared-storage-gpfs2/intern-pretrain-shared02/shijiayang-p/archspace-MoSE/reproduce/olmo-core-backend/scripts/train/FFN_Width_Ablations/run_w1.sh
```

将unique替换为实际唯一标识；W1-QK/W1-M/W2/G1使用各自脚本和唯一job名。launcher沿用既有GPFS2环境、离线W&B、独立编译缓存与checkpoint写盘预检，只在分配到的GPU任务内执行训练。代码不增加论文新颖性承诺：这里检验共享低秩SwiGLU的展开宽度与瓶颈分配，只有可重复的收益成立后才能作为贡献。

本组监控入口：本地运行python scripts/train/FFN_Width_Ablations/monitor.py --loop，每1800秒查看五条job及arch16卡/pool8卡容量。重复启动被文件锁拒绝；排队或SSH超时不触发重提。现场状态保存latest_status.json与monitor_history，实际job ID见jobs.json。
