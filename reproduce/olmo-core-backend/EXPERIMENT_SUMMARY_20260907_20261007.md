# MoSE 实验进展：2026年9月17日之后

统计范围：2026年9月17日—10月7日，按新增实验的首次提交阶段归组；9月17日前实验的续训和同步不计为本期新实验。训练状态截至2026-10-07 16:28 HKT；完整Train CE统计更新至16:36 HKT。

**已有完整Train CE统计的低秩候选中，C6训练最好，C2最终验证最好；两组新增初始化种子重复了这个方向。A1/A2及全部FFN重分配配置均已完成完整CE统计，没有超过C6。** 当前结果说明共享、激活与秩分配需要联合设计，单项收益不能直接相加；低秩配置尚未追平较大参数量的Dense。

## 1. 统计范围、指标与W&B名称

本期共**45条逻辑训练run**，均采用30B训练日程：**45条训练完成**。45条均已完成完整Train CE统计，最后补齐的F4/F5已核实Succeeded、active=0及step14306最终保存。6条新增初始化种子重复单独计数；同一run的中断和续训合并计算。

每条实验的完整W&B名称和链接直接列在下方结果表中。名称是W&B run name；重启attempt归入同一个逻辑run。配套[45条实验CSV](scripts/analysis/Monthly_Summary_20261007/post_20260917_training_inventory.csv)、[JSON](scripts/analysis/Monthly_Summary_20261007/post_20260917_training_inventory.json)及[最新终点/运行状态证据](scripts/analysis/Monthly_Summary_20261007/post_20260917_status_snapshot_20261007T1628.json)采用相同范围。

Dense作为已有质量与参数预算对照，不计入本期45条新run。其W&B名称为[olmo3-1b-dense-baseline-bs256-seq8k-30b-wsd-fp8-5e-3](https://wandb.ai/archspace/MoSE/runs/olmo3-1b-dense-baseline-bs256-seq8k-30b-wsd-fp8-5e-3)；末1000步Train CE为2.357587，参数量1,484,916,736。

MoSE的基本投影为：
\[
a=XA,\qquad z=\phi(a),\qquad Y=zB.
\]

共享指多个分支共用输入因子A，各自保留输出因子B；不是令分支输出相同。Identity表示因子之间没有额外变换；UG中间Linear/Identity仍保留原生SwiGLU的 \(\mathrm{SiLU}(G)\odot U\)。加入SiLU后，整个模块不能直接等同于一个线性低秩矩阵。

主要训练配方为OLMo3-1B骨架：16层、d_model2048、FFN宽8192，序列8192、全局batch256、30B tokens／14306 steps、Muon LR0.005、WSD warmup2000及末20% decay、FP8。新任务每卡microbatch32768 tokens；部分旧参考为65536。主要结果使用**末1000步平均Train CE**，全程平均与原始末步另列；不把W&B平滑值或最后一个batch当作窗口均值。

| 参数预算 | 参数量 | 用途 |
| --- | ---: | --- |
| 所有投影独立rank512 | 797,050,880 | 当前主要组合、共享及秩分配消融 |
| 原生Dense | 1,484,916,736 | 较大模型参考；另做接近其预算的消融 |

两种预算不能混称“同参数量”。当前严格匹配组合每层投影参数均为24,117,248；共享节省的参数通过重新分配rank补回。参数匹配不等于实测吞吐或显存匹配。

## 2. 按主要提交阶段排列的实验概览

阶段根据提交台账归纳，不使用W&B离线同步后的创建时间；同一阶段有多组并行运行。

| 主要提交阶段 | 实验组 | 运行数 | 完成情况 | 主要结果 |
| --- | --- | ---: | --- | --- |
| 9月29日 | Dense预算全投影因子化 | 2 | 全部30B完成 | 不共享F0和共享F1都不如Dense |
| 9月30日—10月2日 | C0–C8组合 | 9 | 全部30B完成 | 严格低秩预算，C6训练最好、C2验证最好 |
| 10月2–3日 | Dense单点SiLU | 4 | 全部30B完成 | K-only末期略优Dense；Up/Gate变差 |
| 10月5日 | Dense预算共享 | 3 | 全部30B完成 | KV最接近Dense，但三组均未超过 |
| 10月6日 | UG部分共享 | 9 | 全部30B完成 | 部分共享没有超过独立端点或C6 |
| 10月6–7日 | FFN秩重分配 | 10 | 全部30B完成，完整CE已统计 | 十组均未超过C6；F3QK最好，接近P0QK |
| 10月6日 | C0/C2/C6种子重复 | 6 | 全部30B完成 | 两组新增种子均支持C6训练最低、C2验证最低 |
| 10月7日 | Attention秩重分配 | 2 | 全部30B完成，完整CE已统计 | A1/A2均差于C6约0.024 |
| **合计** | | **45** | **45训练完成** | **45条均有完整CE统计** |

## 3. Dense预算全投影因子化

本节F0/F1与第8节797M预算的F1/F2不是同一组。Attention中间Identity、UG中间SiLU、Down RMS，参数量均1,484,392,448，比Dense少0.0353%。

| 配置 | 关键rank | 末1000步Train CE | W&B完整实验名称 |
| --- | --- | ---: | --- |
| 原生Dense参考 | 原生dense投影 | **2.357587** | [olmo3-1b-dense-baseline-bs256-seq8k-30b-wsd-fp8-5e-3](https://wandb.ai/archspace/MoSE/runs/olmo3-1b-dense-baseline-bs256-seq8k-30b-wsd-fp8-5e-3) |
| Dense预算F0，不共享 | QKVO1024、UG1640、Down1632 | 2.392441 | [olmo3-1b-densematched-mose-f0-unshared-down-rms-fp8-30b](https://wandb.ai/archspace/MoSE/runs/olmo3-1b-densematched-mose-f0-unshared-down-rms-fp8-30b) |
| Dense预算F1，KV及UG共享 | Q/O1024、KV1360、UG1824、Down1632 | 2.397395 | [olmo3-1b-densematched-mose-f1-kv-gateup-shared-down-rms-fp8-30b](https://wandb.ai/archspace/MoSE/runs/olmo3-1b-densematched-mose-f1-kv-gateup-shared-down-rms-fp8-30b) |

F1没有超过F0，两者都未超过Dense。共享和因子化不是在高参数预算下自动有效的改动。[Dense预算全因子化报告](scripts/train/DenseMatched_MoSE/README.md)。

## 4. 严格低秩预算组合：C0–C8

9条均已完成30B训练，总参数量均为**797,050,880**，每层投影参数均为**24,117,248**，严格匹配所有投影独立rank512的预算。训练日程沿用第1节，不与更大参数量的Dense混作同预算比较。

### 4.1 完整结构配置

每个投影单元格写为“**rank / 低秩因子之间的变换**”，对应 \(Y=\phi(XA)B\)；Linear即Identity，表示 \(\phi(a)=a\)。rank是中间通道维度。Down的RMSNorm均沿该中间维度归一化，**没有可学习gain**。

| 配置 | Q：rank / 变换 | K：rank / 变换 | V：rank / 变换 | O：rank / 变换 | Up：rank / 变换 | Gate：rank / 变换 | Down：rank / 变换 | K/V输入因子共享 | Up/Gate输入因子共享 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C0 | 512 / Linear | 512 / SiLU | 512 / Linear | 512 / Linear | 512 / Linear | 512 / Linear | 512 / RMSNorm | 否 | 否 |
| C1 | 512 / Linear | 688 / SiLU | 688 / Linear | 504 / Linear | 512 / Linear | 512 / Linear | 512 / RMSNorm | 是 | 否 |
| C2 | 512 / Linear | 512 / SiLU | 512 / Linear | 496 / Linear | 568 / Linear | 568 / Linear | 520 / RMSNorm | 否 | 是 |
| C3 | 520 / Linear | 680 / SiLU | 680 / Linear | 512 / Linear | 568 / Linear | 568 / Linear | 512 / RMSNorm | 是 | 是 |
| C4 | 512 / SiLU | 688 / SiLU | 688 / SiLU | 504 / SiLU | 512 / Linear | 512 / Linear | 512 / RMSNorm | 是 | 否 |
| C5 | 512 / SiLU | 512 / Linear | 512 / Linear | 496 / Linear | 568 / Linear | 568 / Linear | 520 / RMSNorm | 否 | 是 |
| C6 | 512 / SiLU | 512 / SiLU | 512 / Linear | 496 / Linear | 568 / Linear | 568 / Linear | 520 / RMSNorm | 否 | 是 |
| C7 | 512 / SiLU | 512 / SiLU | 512 / Linear | 496 / Linear | 568 / SiLU | 568 / SiLU | 520 / RMSNorm | 否 | 是 |
| C8 | 512 / Linear | 512 / Linear | 512 / Linear | 496 / Linear | 568 / Linear | 568 / Linear | 520 / RMSNorm | 否 | 是 |

共享只发生在同层的输入因子A，输出因子B始终分支独立。所有配置的Q与K都不共享输入因子；“Q/K SiLU”只描述两个位置的变换。C1/C3中K、V虽然共用输入因子，但K用SiLU、V用Linear，激活后的latent不同。为补回共享节省的参数，表中部分O、Down及Q的rank作了调整。

### 4.2 投影与FFN公式

令 \(X_a\) 为Attention分支输入，\(H_a\) 为注意力聚合后的O投影输入，\(X_f\) 为FFN分支输入。这里只展开投影，位置编码及外围归一化按基线保留：

\[
Q=\phi_Q(X_aA_Q)B_Q,\qquad
K=\phi_K(X_aA_K)B_K,\qquad
V=\phi_V(X_aA_V)B_V,\qquad
Y_a=\phi_O(H_aA_O)B_O.
\]

\[
U=\phi_U(X_fA_U)B_U,\qquad
G=\phi_G(X_fA_G)B_G,\qquad
H_f=\mathrm{SiLU}(G)\odot U,\qquad
Y_f=\mathrm{RMSNorm}(H_fA_D)B_D.
\]

表中的Up/Gate共享对应 \(A_U=A_G=A_{UG}\)，K/V共享对应 \(A_K=A_V=A_{KV}\)，各位置的 \(\phi\) 按表取值。**所有配置均保留原生SwiGLU的 \(\mathrm{SiLU}(G)\odot U\)**；Up/Gate中间Linear和C8的“全部Linear”均不表示删除这一步。C7是在该原生门控之外，还对Up/Gate低秩latent额外施加SiLU。

### 4.3 Train CE与完整W&B名称

主要比较末期窗口平均Train CE，越低越好。统一排除steps12516、14304：全程均值采用14,304个共同有效点，末1000步窗口steps13307–14306采用999个共同有效点；下表没有使用原始最后一步或W&B平滑值。

| 配置 | 全程平均Train CE | 末期窗口平均Train CE | W&B完整实验名称 |
| --- | ---: | ---: | --- |
| C0 | 2.793340 | 2.479042 | [olmo3-1b-r512matched-c0-k-silu-down-rms-fp8-30b](https://wandb.ai/archspace/MoSE/runs/olmo3-1b-r512matched-c0-k-silu-down-rms-fp8-30b) |
| C1 | 2.797321 | 2.481070 | [olmo3-1b-r512matched-c1-k-silu-down-rms-kv-fp8-30b](https://wandb.ai/archspace/MoSE/runs/olmo3-1b-r512matched-c1-k-silu-down-rms-kv-fp8-30b) |
| C2 | 2.791325 | 2.475732 | [olmo3-1b-r512matched-c2-k-silu-down-rms-gateup-fp8-30b](https://wandb.ai/archspace/MoSE/runs/olmo3-1b-r512matched-c2-k-silu-down-rms-gateup-fp8-30b) |
| C3 | 2.796868 | 2.478402 | [olmo3-1b-r512matched-c3-k-silu-down-rms-kv-gateup-fp8-30b](https://wandb.ai/archspace/MoSE/runs/olmo3-1b-r512matched-c3-k-silu-down-rms-kv-gateup-fp8-30b) |
| C4 | 2.798778 | 2.479408 | [olmo3-1b-r512matched-c4-attention-silu-down-rms-kv-fp8-30b](https://wandb.ai/archspace/MoSE/runs/olmo3-1b-r512matched-c4-attention-silu-down-rms-kv-fp8-30b) |
| C5 | 2.791086 | 2.475784 | [olmo3-1b-r512matched-c5-q-silu-down-rms-gateup-fp8-30b](https://wandb.ai/archspace/MoSE/runs/olmo3-1b-r512matched-c5-q-silu-down-rms-gateup-fp8-30b) |
| C6 | 2.789434 | 2.474827 | [olmo3-1b-r512matched-c6-qk-silu-down-rms-gateup-fp8-30b](https://wandb.ai/archspace/MoSE/runs/olmo3-1b-r512matched-c6-qk-silu-down-rms-gateup-fp8-30b) |
| C7 | 2.808255 | 2.492346 | [olmo3-1b-r512matched-c7-qk-silu-ug-silu-down-rms-gateup-fp8-30b](https://wandb.ai/archspace/MoSE/runs/olmo3-1b-r512matched-c7-qk-silu-ug-silu-down-rms-gateup-fp8-30b) |
| C8 | 2.790090 | 2.475791 | [olmo3-1b-r512matched-c8-linear-down-rms-gateup-fp8-30b](https://wandb.ai/archspace/MoSE/runs/olmo3-1b-r512matched-c8-linear-down-rms-gateup-fp8-30b) |

C2相对C0有改善，C1相对C0更差，C3也差于C2；UG共享与秩重分配的收益不能直接再叠加KV共享。C6只比C2低约0.000905；C8与C2近乎持平。C7比C6差约0.017519，支持UG中间保持Linear。C2对C0同时改变共享和rank，不能把全部收益单独归因于共享。[C0–C7](scripts/train/Rank512Matched_Combinations/C0_C7_RESULTS.md)／[C8](scripts/train/Rank512Matched_Combinations/C8_RESULTS.md)／[精确配置表](scripts/train/Rank512Matched_Combinations/README.md)。

在新增20配置联合比较中只共同排除step12516，因此C6末窗为1000个点、CE为**2.474763**；与上表2.474827的微小差异来自共同异常掩码。它不是另一条C6运行。[联合比较](scripts/train/UG_Partial_Sharing/P3_P4_QK_RESULTS.md)。

## 5. Dense单点SiLU

只因子化选定的Q/K/Up/Gate，其他仍为Dense，Down保持原生Dense。

| 配置 | 参数量 | 全程平均Train CE | 末1000步Train CE | W&B完整实验名称 |
| --- | ---: | ---: | ---: | --- |
| Dense | 1,484,916,736 | 2.664923 | 2.357587 | [olmo3-1b-dense-baseline-bs256-seq8k-30b-wsd-fp8-5e-3](https://wandb.ai/archspace/MoSE/runs/olmo3-1b-dense-baseline-bs256-seq8k-30b-wsd-fp8-5e-3) |
| Q-only，rank1024 | 1,484,916,736 | 2.667829 | 2.357467 | [olmo3-1b-dense-matched-q-only-silu-r1024-fp8-30b](https://wandb.ai/archspace/MoSE/runs/olmo3-1b-dense-matched-q-only-silu-r1024-fp8-30b) |
| K-only，rank1024 | 1,484,916,736 | 2.665999 | 2.356372 | [olmo3-1b-dense-matched-k-only-silu-r1024-fp8-30b](https://wandb.ai/archspace/MoSE/runs/olmo3-1b-dense-matched-k-only-silu-r1024-fp8-30b) |
| Up-only，rank1632 | 1,483,868,160 | 2.672787 | 2.363876 | [olmo3-1b-dense-nearmatched-up-only-silu-r1632-fp8-30b](https://wandb.ai/archspace/MoSE/runs/olmo3-1b-dense-nearmatched-up-only-silu-r1632-fp8-30b) |
| Gate-only，rank1632 | 1,483,868,160 | 2.672209 | 2.361813 | [olmo3-1b-dense-nearmatched-gate-only-silu-r1632-fp8-30b](https://wandb.ai/archspace/MoSE/runs/olmo3-1b-dense-nearmatched-gate-only-silu-r1632-fp8-30b) |

K-only末期比Dense低约0.001215，Q-only近乎持平；全程平均仍由Dense领先。Up/Gate额外SiLU更差。新run与旧Dense的microbatch及因子初始化有差异，不能宣称已获得稳定整体提升。[单点替换结果](scripts/train/Dense_Single_SiLU/RESULTS.md)。

## 6. Dense预算共享

共享latent中间Linear；Q/O、Down保持原生Dense，UG保留原生SwiGLU。

| 配置 | 共享rank | 参数量 | 末1000步Train CE | W&B完整实验名称 |
| --- | --- | ---: | ---: | --- |
| Dense | 无 | 1,484,916,736 | 2.357587 | [olmo3-1b-dense-baseline-bs256-seq8k-30b-wsd-fp8-5e-3](https://wandb.ai/archspace/MoSE/runs/olmo3-1b-dense-baseline-bs256-seq8k-30b-wsd-fp8-5e-3) |
| S1，UG共享 | 1808 | 1,481,246,720 | 2.368455 | [olmo3-1b-dense-sharing-s1-linear-fp8-30b](https://wandb.ai/archspace/MoSE/runs/olmo3-1b-dense-sharing-s1-linear-fp8-30b) |
| S2，KV共享 | 1360 | 1,484,392,448 | 2.358965 | [olmo3-1b-dense-sharing-s2-linear-fp8-30b](https://wandb.ai/archspace/MoSE/runs/olmo3-1b-dense-sharing-s2-linear-fp8-30b) |
| S3，UG+KV共享 | 1808/1360 | 1,480,722,432 | 2.371111 | [olmo3-1b-dense-sharing-s3-linear-fp8-30b](https://wandb.ai/archspace/MoSE/runs/olmo3-1b-dense-sharing-s3-linear-fp8-30b) |

S2最接近Dense，但三者在全程均值和末期均值都未超过。向下取整导致参数略少，必须披露；这种对照同时改变因子化、共享及初始化，不是纯共享效应。[Dense共享结果](scripts/train/Dense_Sharing/RESULTS.md)。

## 7. UG部分共享

P组Attention及Down均为独立rank512，Down RMS，UG中间Linear。qk版只在独立Q/K中间加SiLU，不表示Q/K输入因子共享。

| 配置 | UG共享rank | 每分支私有rank | Q/K中间变换 | 末1000步Train CE | W&B完整实验名称 |
| --- | ---: | ---: | --- | ---: | --- |
| P0 | 0 | 512 | Identity | 2.480162 | [olmo3-1b-ug-partial-p0-s0-p512-linear-down-rms-fp8-30b](https://wandb.ai/archspace/MoSE/runs/olmo3-1b-ug-partial-p0-s0-p512-linear-down-rms-fp8-30b) |
| P0QK | 0 | 512 | SiLU | 2.477410 | [olmo3-1b-ug-partial-p0qk-s0-p512-linear-qk-silu-down-rms-fp8-30b](https://wandb.ai/archspace/MoSE/runs/olmo3-1b-ug-partial-p0qk-s0-p512-linear-qk-silu-down-rms-fp8-30b) |
| P1 | 560 | 0 | Identity | 2.483417 | [olmo3-1b-ug-partial-p1-s560-p0-linear-down-rms-fp8-30b](https://wandb.ai/archspace/MoSE/runs/olmo3-1b-ug-partial-p1-s560-p0-linear-down-rms-fp8-30b) |
| P2 | 480 | 80 | Identity | 2.484497 | [olmo3-1b-ug-partial-p2-s480-p80-linear-down-rms-fp8-30b](https://wandb.ai/archspace/MoSE/runs/olmo3-1b-ug-partial-p2-s480-p80-linear-down-rms-fp8-30b) |
| P2QK | 480 | 80 | SiLU | 2.481273 | [olmo3-1b-ug-partial-p2qk-s480-p80-linear-qk-silu-down-rms-fp8-30b](https://wandb.ai/archspace/MoSE/runs/olmo3-1b-ug-partial-p2qk-s480-p80-linear-qk-silu-down-rms-fp8-30b) |
| P3 | 320 | 224 | Identity | 2.484301 | [olmo3-1b-ug-partial-p3-s320-p224-linear-down-rms-fp8-30b](https://wandb.ai/archspace/MoSE/runs/olmo3-1b-ug-partial-p3-s320-p224-linear-down-rms-fp8-30b) |
| P3QK | 320 | 224 | SiLU | 2.481997 | [olmo3-1b-ug-partial-p3qk-s320-p224-linear-qk-silu-down-rms-fp8-30b](https://wandb.ai/archspace/MoSE/runs/olmo3-1b-ug-partial-p3qk-s320-p224-linear-qk-silu-down-rms-fp8-30b) |
| P4 | 160 | 368 | Identity | 2.485872 | [olmo3-1b-ug-partial-p4-s160-p368-linear-down-rms-fp8-30b](https://wandb.ai/archspace/MoSE/runs/olmo3-1b-ug-partial-p4-s160-p368-linear-down-rms-fp8-30b) |
| P4QK | 160 | 368 | SiLU | 2.481281 | [olmo3-1b-ug-partial-p4qk-s160-p368-linear-qk-silu-down-rms-fp8-30b](https://wandb.ai/archspace/MoSE/runs/olmo3-1b-ug-partial-p4qk-s160-p368-linear-qk-silu-down-rms-fp8-30b) |

P1为794,429,440参数，比目标少0.329%，单列；其余精确匹配797,050,880。部分共享内部点没有超过独立端点P0/P0qk，也未超过C6。四个背景中Q/K SiLU均改善末期CE，但P2有过32→16卡恢复及rank RNG变化，硬件/恢复差异保留。P组与FFN组在20配置联合比较中共同排除step12516，末期窗口1000点。[配置和公式](scripts/train/UG_Partial_Sharing/README.md)／[完整结果](scripts/train/UG_Partial_Sharing/P3_P4_QK_RESULTS.md)。

**“独立端点最好”仅限P组已测试的设置，不能扩展为“UG不共享普遍优于全共享”。** C组中C2相对C0的UG全共享与秩重分配组合仍有收益，两个新增种子的末期Train CE平均改善0.003664。P组固定Q/K/V/O与Down为512；C2/C6/C8则使用UG共享568、O496、Down520。P1的UG全共享rank仅560，参数略少；Q/K SiLU背景下还缺少P1QK全共享端点。因此两组不是同一个共享比例消融。C8同样采用Attention/UG中间Linear，其结果也优于P0，说明不能仅凭激活背景解释方向差异。当前可以确认部分共享未带来增益，但全共享端点的反向结果仍需统一配置、实现与初始化后核查；参数、秩或训练恢复差异尚未被单独证明为原因。

## 8. FFN秩重分配

Attention独立rank512，UG中间Linear，Down无gain RMS。这里F1–F5为797M预算，与第3节Dense预算F0/F1属于不同实验。

| 配置 | UG rank | UG共享 | Down rank | Q/K中间变换 | 末1000步Train CE | W&B完整实验名称 |
| --- | ---: | --- | ---: | --- | --- | --- |
| F1 | 384 | 否 | 768 | Identity | 2.482142 | [olmo3-1b-ffn-budget-f1-ug384-down768-unshared-linear-rms-fp8-30b](https://wandb.ai/archspace/MoSE/runs/olmo3-1b-ffn-budget-f1-ug384-down768-unshared-linear-rms-fp8-30b) |
| F1QK | 384 | 否 | 768 | SiLU | 2.483381 | [olmo3-1b-ffn-budget-f1qk-ug384-down768-unshared-qk-silu-ug-linear-down-rms-fp8-30b](https://wandb.ai/archspace/MoSE/runs/olmo3-1b-ffn-budget-f1qk-ug384-down768-unshared-qk-silu-ug-linear-down-rms-fp8-30b) |
| F2 | 256 | 否 | 1024 | Identity | 2.498586 | [olmo3-1b-ffn-budget-f2-ug256-down1024-unshared-linear-rms-fp8-30b](https://wandb.ai/archspace/MoSE/runs/olmo3-1b-ffn-budget-f2-ug256-down1024-unshared-linear-rms-fp8-30b) |
| F2QK | 256 | 否 | 1024 | SiLU | 2.494263 | [olmo3-1b-ffn-budget-f2qk-ug256-down1024-unshared-qk-silu-ug-linear-down-rms-fp8-30b](https://wandb.ai/archspace/MoSE/runs/olmo3-1b-ffn-budget-f2qk-ug256-down1024-unshared-qk-silu-ug-linear-down-rms-fp8-30b) |
| F3 | 480 | 是 | 672 | Identity | 2.481782 | [olmo3-1b-ffn-budget-f3-ug480-down672-shared-linear-rms-fp8-30b](https://wandb.ai/archspace/MoSE/runs/olmo3-1b-ffn-budget-f3-ug480-down672-shared-linear-rms-fp8-30b) |
| F3QK | 480 | 是 | 672 | SiLU | 2.477421 | [olmo3-1b-ffn-budget-f3qk-ug480-down672-shared-qk-silu-ug-linear-down-rms-fp8-30b](https://wandb.ai/archspace/MoSE/runs/olmo3-1b-ffn-budget-f3qk-ug480-down672-shared-qk-silu-ug-linear-down-rms-fp8-30b) |
| F4 | 400 | 是 | 816 | Identity | 2.486620 | [olmo3-1b-ffn-budget-f4-ug400-down816-shared-linear-rms-fp8-30b](https://wandb.ai/archspace/MoSE/runs/olmo3-1b-ffn-budget-f4-ug400-down816-shared-linear-rms-fp8-30b) |
| F4QK | 400 | 是 | 816 | SiLU | 2.485783 | [olmo3-1b-ffn-budget-f4qk-ug400-down816-shared-qk-silu-ug-linear-down-rms-fp8-30b](https://wandb.ai/archspace/MoSE/runs/olmo3-1b-ffn-budget-f4qk-ug400-down816-shared-qk-silu-ug-linear-down-rms-fp8-30b) |
| F5 | 560 | 是 | 528 | Identity | 2.484029 | [olmo3-1b-ffn-budget-f5-ug560-down528-shared-linear-rms-fp8-30b](https://wandb.ai/archspace/MoSE/runs/olmo3-1b-ffn-budget-f5-ug560-down528-shared-linear-rms-fp8-30b) |
| F5QK | 560 | 是 | 528 | SiLU | 2.479613 | [olmo3-1b-ffn-budget-f5qk-ug560-down528-shared-qk-silu-ug-linear-down-rms-fp8-30b](https://wandb.ai/archspace/MoSE/runs/olmo3-1b-ffn-budget-f5qk-ug560-down528-shared-qk-silu-ug-linear-down-rms-fp8-30b) |

十组均已完成30B并统计完整Train CE，全部严格797,050,880参数。固定Attention各投影rank512、仅在FFN内部重分配时，**F3QK的UG共享rank480／Down rank672最好，末1000步CE为2.477421**；与不共享UG512／Down512的P0QK（2.477410）近乎持平，仍比C6（2.474763）高0.002658。大幅减少UG、增加Down的F2明显不利；补齐的F4、F5末期CE为2.486620、2.484029，也未获得更优配置。Q/K SiLU在F1末期反而恶化约0.001238，说明激活收益依赖容量分配，不能简单叠加。

F4/F5的精确W&B段已确认Finished syncing，API无采样历史均完整覆盖steps1–14306；重新读取的C0/C2/C6与此前联合结果一致。十组按共同异常step12516排除后的全程14305点、末窗1000点比较，原始记录与SHA256见[补齐历史](scripts/train/FFN_Rank_Reallocation/analysis/F4_F5_WITH_REFERENCES_FULL_TRAIN_CE.json)，全部十组、独立端点和C组参考见[统一统计](scripts/train/FFN_Rank_Reallocation/analysis/FFN_ALL_WITH_REFERENCES_FULL_TRAIN_CE.json)。当前结论是这批重分配没有改善最优质量；它支持避免过度削减UG，不能证明某个rank为全局最优。

**若允许全模型的rank一起调整，同参数量下仍以C6的UG共享568／Down520最好。** C6还将O从512减为496（Q/K/V仍为512），给FFN多出每层65,536参数；因此它与本节固定Attention512的FFN预算曲线有不同约束。UG维度均指Up/Gate共用输入因子的宽度，各分支输出因子独立；Down均无gain RMSNorm。[配置](scripts/train/FFN_Rank_Reallocation/README.md)／[此前联合结果](scripts/train/UG_Partial_Sharing/P3_P4_QK_RESULTS.md)。

## 9. C0/C2/C6两组初始化种子重复

C0/C2/C6各补初始化seed12537/12538，固定data seed34521，6条完整30B训练；均pool8卡、严格797M预算。下表仅平均两个新增种子，不混入原始12536。

| 配置 | 两种子全程平均Train CE | 两种子末1000步Train CE | 修正后最终验证macro CE |
| --- | ---: | ---: | ---: |
| C0 | 2.793230 | 2.479432 | 2.740639 |
| C2 | 2.791133 | 2.475767 | **2.733280** |
| C6 | **2.790045** | **2.475344** | 2.734584 |

两个新增种子中，C2/C6在训练和验证上均优于C0；C6相对C2只平均改善Train CE约0.000423，最终验证反而高0.001304。C6与C2的模型配置只在Q的中间SiLU上不同，取舍方向得到重复，但两个固定数据顺序的初始化重复不足以声称统计显著。[六次完整结果](scripts/train/Rank512_Seed_Replication/C0_C2_C6_RESULTS.md)。

六次运行的完整W&B名称与逐run指标：

| 初始化seed | 配置 | 全程平均Train CE | 末1000步平均Train CE | 最后一条原始Train CE | 最终验证macro CE | W&B完整实验名称 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 12537 | C0 | 2.793364229 | 2.479495967 | 2.509505987 | 2.742605827 | [olmo3-1b-r512matched-c0-seed12537-data34521-fp8-30b](https://wandb.ai/archspace/MoSE/runs/olmo3-1b-r512matched-c0-seed12537-data34521-fp8-30b) |
| 12537 | C2 | 2.790961065 | 2.475723318 | 2.505619049 | 2.733086066 | [olmo3-1b-r512matched-c2-seed12537-data34521-fp8-30b](https://wandb.ai/archspace/MoSE/runs/olmo3-1b-r512matched-c2-seed12537-data34521-fp8-30b) |
| 12537 | C6 | 2.790239032 | 2.475591339 | 2.505630732 | 2.734566862 | [olmo3-1b-r512matched-c6-seed12537-data34521-fp8-30b](https://wandb.ai/archspace/MoSE/runs/olmo3-1b-r512matched-c6-seed12537-data34521-fp8-30b) |
| 12538 | C0 | 2.793095821 | 2.479367399 | 2.508933544 | 2.738672679 | [olmo3-1b-r512matched-c0-seed12538-data34521-fp8-30b](https://wandb.ai/archspace/MoSE/runs/olmo3-1b-r512matched-c0-seed12538-data34521-fp8-30b) |
| 12538 | C2 | 2.791305917 | 2.475811098 | 2.505337000 | 2.733473084 | [olmo3-1b-r512matched-c2-seed12538-data34521-fp8-30b](https://wandb.ai/archspace/MoSE/runs/olmo3-1b-r512matched-c2-seed12538-data34521-fp8-30b) |
| 12538 | C6 | 2.789851389 | 2.475097591 | 2.505316973 | 2.734601259 | [olmo3-1b-r512matched-c6-seed12538-data34521-fp8-30b](https://wandb.ai/archspace/MoSE/runs/olmo3-1b-r512matched-c6-seed12538-data34521-fp8-30b) |

![两组初始化种子的C0/C2/C6训练与验证比较](scripts/analysis/Corrected_LM_Evaluation/figures/paired_seed_c0_c2_c6.png)

## 10. Attention秩重分配

本组把一部分FFN预算转给Attention，两个配置均严格797,050,880参数：Q/K SiLU，V/O和UG中间Linear，Down无gain RMS，Attention不共享。

| 配置 | Q/K/V/O rank | UG rank | UG共享 | Down rank | 全程平均Train CE | 末1000步Train CE | 末期相对C6 | W&B完整实验名称 |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| A1 | 832 | 336 | 否 | 352 | 2.806921 | 2.498468 | +0.023705 | [olmo3-1b-attn-budget-a1-r832-ug336-down352-unshared-qk-silu-down-rms-fp8-30b](https://wandb.ai/archspace/MoSE/runs/olmo3-1b-attn-budget-a1-r832-ug336-down352-unshared-qk-silu-down-rms-fp8-30b) |
| A2 | 832 | 320 | 是 | 448 | 2.808386 | 2.498580 | +0.023817 | [olmo3-1b-attn-budget-a2-r832-ug320-down448-shared-qk-silu-down-rms-fp8-30b](https://wandb.ai/archspace/MoSE/runs/olmo3-1b-attn-budget-a2-r832-ug320-down448-shared-qk-silu-down-rms-fp8-30b) |

A2于10月7日13:46:46完成，A1于13:49:52完成；均已核实Succeeded、active=0/succeeded=1及step14306最终保存，精确离线W&B段的Finished syncing已确认。通过W&B API的无采样scan_history重新读取A1/A2及C0/C2/C6，五组均完整覆盖steps1–14306；共同排除C6异常step12516，全程14305点，末期窗口13307–14306完整1000点。原始记录、SHA256和统一统计保存在[完整CE数据](scripts/train/Attention_Rank_Reallocation/analysis/A1_A2_WITH_REFERENCES_FULL_TRAIN_CE.json)。

同口径C6末期CE为2.474763，C2为2.475668，C0为2.478978；**A1/A2均未改善质量，末期比C6高约0.024，全程均值也更差。** 两个新增预算点的末期差仅约0.000112，应视为接近。当前结果不支持通过大幅压缩FFN、提高Attention至832获得更优配置；这同时改变两个模块的容量，不能单独证明Attention加秩没有价值。A2对A1还改变共享及FFN秩，不把差异全部归因于共享。这是初始化seed12536、数据seed34521的筛选，没有新增统一最终验证结论。[配置](scripts/train/Attention_Rank_Reallocation/README.md)／[完成证据](scripts/analysis/Monthly_Summary_20261007/post_20260917_status_snapshot_20261007T1628.json)。

## 11. 本期独立重评、完整历史分析与效率微基准

本期独立分析关联上表源模型，没有新增训练W&B run，不计入45条训练run。

| 已运行分析 | 覆盖范围 | 结果与用途 |
| --- | --- | --- |
| 完整Train CE导出与联合比较 | C组、P组、F组、A组、Dense组及种子重复 | 对齐step/tokens，合并续训段，保留异常点及共同掩码 |
| 修正mask后的最终LM重评 | 7个原始低秩配置、Dense、6条新增种子重复 | 同step14306、同11源与有效目标标签；C2验证最优 |
| 同节点训练步效率微基准 | Dense/C0/C6，同一H200节点8卡 | C6吞吐1.235×Dense，但PyTorch峰值allocated显存更高 |


旧训练内LM评估存在label shift后的有效位置与未shift mask错位。独立重评按真正有效目标标签评分；Train CE不受这个评估实现问题影响。原始C0/C2/C6修正后最终验证为2.738866／**2.732681**／2.734855；Dense为2.627212。它们不能与旧mask、step14000的验证绝对值混用。[统一重评](scripts/analysis/Corrected_LM_Evaluation/RESULTS.md)／[Dense重评](scripts/analysis/Corrected_LM_Evaluation/DENSE_COMPARISON.md)。

| 同节点训练步测量 | 平均训练步秒数 | 全局tokens/s | 峰值allocated GiB |
| --- | ---: | ---: | ---: |
| Dense | 3.690175 | 568,307 | 51.269 |
| C0 | 2.901662 | 722,742 | 56.488 |
| C6 | 2.987917 | 701,878 | 66.305 |

这是固定合成token、真实forward/backward/Muon更新的训练步微基准：5步预热、10步测量，统一microbatch32768；不包含数据加载、eval和checkpoint开销。不能把它称为完整训练加速，也不能把参数减少46.3%写成显存同比节省。[效率报告](scripts/analysis/Training_Efficiency/RESULTS.md)。

## 12. 当前选型与本期结论

已有完整训练统计的候选中，按主要Train CE选择C6；修正后最终验证保留C2为优选。C6的完整配置为Q512/K512+SiLU、V512/O496+Linear；UG共享rank568、中间Linear；Down520+无gain RMSNorm。Attention不共享，原生SwiGLU及外围归一化、RoPE保留。C2仅将Q中间SiLU改为Identity。

本期可以支持的主线是：**按投影功能选择低秩中间变换，联合设计共享输入空间与固定参数预算下的秩分配。** C组支持UG共享与预算重分配的组合，但继续叠加KV共享、UG额外SiLU、部分共享或过度减少UG容量都未带来更好的已统计终点；两个新增种子重复了C2/C6相对C0改善的方向。新增种子中，额外Q SiLU仅平均改善Train CE约0.000423，最终验证反而高0.001304，训练与验证取舍必须同时呈现。

原始C6的末期Train CE约2.474763，较大Dense为2.357587，仍有约0.117176差距；C6参数减少约46.3%，尚不能宣称追平Dense。A1/A2及十组FFN重分配的完整统计均未超过C6。主配置仍需统一下游质量评估与完整训练效率。低秩分解、共享和RMSNorm本身不作为新发明，贡献落在结构组合、受控实验发现及可复现证据上。

[项目概况](PROJECT_OVERVIEW.md)。
