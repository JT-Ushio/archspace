# MoSE 实验进度交接稿

这份说明面向第一次接触 MoSE、需要代为汇报实验进度的人。汇报时先讲清研究问题，再讲实验结果；不要从单条 W&B 曲线开始讲。

## 1. 一句话说明我们在做什么

我们以 OLMo3-1B 为训练骨架，把 Transformer 中的大线性映射拆成低秩形式，并研究：

1. 不同投影能否共享同一个低维输入子空间；
2. 低维空间中应该使用 SiLU、RMSNorm，还是不加非线性；
3. SVD 初始化、输出子空间旋转等方法能否进一步改善低秩模型。

核心目标是在参数量和训练预算可比的条件下，判断哪些结构约束真正有效。主指标是交叉熵 CE，**越低越好**。

## 2. MoSE 的最小背景

普通线性层是

\[
y=xW.
\]

MoSE 将它写成一个低维编码和一个输出解码：

\[
a=xA,\qquad z=\phi(a),\qquad y=zB,
\]

其中 \(A\in\mathbb R^{d_{in}\times r}\)、\(B\in\mathbb R^{r\times d_{out}}\)，且 \(r\ll d_{in},d_{out}\)。

- \(a\)：进入中间变换前的 rank-\(r\) latent；
- \(z\)：经过 SiLU、RMSNorm 或 Identity 后的 latent；
- \(A\)：决定从宽残差流中提取什么低维信息；
- \(B\)：将低维信息读出为 Q、K、V、O 或 FFN 输出。

“共享”的含义是多个分支共用 \(A\)，但保留各自的 \(B\)。例如 Q/K 共享：

\[
y_q=\phi(xA_{qk})B_q,\qquad
y_k=\phi(xA_{qk})B_k.
\]

它不是令 Q 和 K 完全相同。二者读取同一份 latent，但可以用不同的 \(B_q,B_k\) 解释它。

本项目中常说的 baseline 可能有两种含义，汇报时要说明：

- **MoSE unshared baseline**：所有映射都已低秩化，但 Q/K/V/O、Gate/Up 各有自己的输入投影；
- **原生 OLMo baseline**：没有这套低秩 MoSE 分解的 dense 模型。

当前多数消融比较的是前一种 baseline。

## 3. 公平比较：为什么叫同参数量消融

共享一个 \(A\) 会自然减少参数。如果仍固定 rank=512，共享模型会因为参数更少而处于不公平条件。因此参数匹配实验把节省下来的参数用于增加 latent rank：

| 结构 | 共享方式 | 参数匹配 rank |
| --- | --- | ---: |
| baseline | 不共享 | 512 |
| Gate/Up 共享 | 共用 \(A_{ug}\) | 568 |
| Q/K 共享 | 共用 \(A_{qk}\) | 680 |
| K/V 共享 | 共用 \(A_{kv}\) | 680 |
| Q/K/V 共享 | 共用 \(A_{qkv}\) | 768 |

这些 rank 是根据各分支输入、输出维度算出的近似同参数量配置。其他关键训练条件保持一致：相同数据顺序、global batch、训练 token、优化器、学习率和调度。参数量相同不表示 FLOPs、通信量或实际吞吐完全相同，因此效率必须单独报告。

## 4. 主结果一：哪些投影适合共享

当前单 seed 的 loss 排序为：

\[
\boxed{L_{kv}\approx L_{ug}<L_{baseline}\approx L_{ug+qk}<L_{qk}<L_{qkv}}
\]

CE 越低越好，因此可以这样解释：

1. **K/V 共享和 Gate/Up 共享最好。** 这些分支接收同一个输入，并且功能上具有较强耦合，共享低维信息提取器是合理的。
2. **Q/K 共享单独看更差。** Q 表示当前查询，K 表示被检索内容；强迫两者先进入同一个 latent，可能限制了它们学习不同坐标系的能力。
3. **Q/K/V 全共享最差。** 共享范围扩大后约束过强，即使将 rank 增至 768，也没有补回结构自由度的损失。
4. **Gate/Up 与 Q/K 同时共享没有叠加两个改动的收益。** 说明“各自可接受的共享”不能假定组合后仍独立有效。

这组实验回答的是“哪些分支可以共用输入子空间”。当前是单 seed 结果，适合说“趋势明确”，不适合说已经证明统计显著。

## 5. 主结果二：Down latent 中究竟需要什么变换

Down 分支是

\[
h\in\mathbb R^{8192}
\xrightarrow{A_d}
a\in\mathbb R^{512}
\xrightarrow{\phi}
z\in\mathbb R^{512}
\xrightarrow{B_d}
y\in\mathbb R^{2048}.
\]

四组只改变 \(\phi\)：

| 简写 | 中间变换 |
| --- | --- |
| S | \(z=\operatorname{SiLU}(a)\) |
| I | \(z=a\) |
| R | \(z=\operatorname{RMSNorm}(a)\) |
| SR | \(z=\operatorname{RMSNorm}(\operatorname{SiLU}(a))\) |

30B 训练的最后共同验证结果为：

| 变体 | Validation macro CE |
| --- | ---: |
| S：SiLU | 2.776709 |
| I：Identity | 2.756521 |
| R：RMSNorm | **2.756267** |
| SR：SiLU 后 RMSNorm | 2.778165 |

因此：

\[
\boxed{L_I\approx L_R<L_S\approx L_{SR}}
\]

从 SiLU 改为 Identity 改善 0.020188 CE；在 Identity 上再加入 RMSNorm 只改善 0.000254。按这一次观测差值计算，SiLU→RMSNorm 的收益中约 98.8% 已经由“移除 SiLU”解释。

目前最合适的结论是：**Down-RMSNorm 表现好，主要因为它没有像 SiLU 那样扭曲 latent 的方向，而不是 RMSNorm 本身提供了很大的额外收益。** SiLU 后再加 RMSNorm 没有恢复性能，因为 RMSNorm只能重新调整整体尺度，无法撤销 SiLU 对各通道相对比例的改变。

## 6. a/z 热力图在说明什么

对每个 token，都有一个 512 维的 \(a_t\) 和 \(z_t\)。分析先分别去掉它们的 RMS 尺度，再比较方向和通道结构：

\[
\hat a_t=\frac{a_t}{\operatorname{RMS}(a_t)},\qquad
\hat z_t=\frac{z_t}{\operatorname{RMS}(z_t)}.
\]

第 8 层统计为：

| 变换 | 平均绝对变化 | \(a/z\) 平均余弦 |
| --- | ---: | ---: |
| SiLU | 0.405939 | 0.826189 |
| RMSNorm | 0.001114 | 0.999999 |

SiLU 会逐维压制负值并非线性调整正值，因此即使去掉幅度影响，\(a\) 与 \(z\) 的方向仍明显不同。无可学习 gain 的 RMSNorm 对一个 token 主要乘以同一个正标量，所以方向几乎不变。RMSNorm 的差值图接近全白是预期结果，不是缺少数据。

这张图提供了与 loss 结果一致的机制线索，但不能单独证明因果关系。两个独立训练模型中相同编号的通道不具有可直接对齐的语义。

## 7. FP8 是否放大了 RMSNorm 的优势

为检验“RMSNorm 只是修复 FP8 数值误差”这一猜想，做了 BF16/FP8 × SiLU/RMSNorm 的 2×2 实验。这组使用 7.5B token，只是完整 30B 预算的四分之一，调度也属于早期专项设置。

最后 10% train CE 中：

- BF16 下 SiLU−RMSNorm：0.022883；
- FP8 下 SiLU−RMSNorm：0.025144；
- 二者差值：0.002261。

最后共同验证的对应收益为 0.024302、0.026494，差值 0.002192。

所以 RMSNorm 在 BF16 和 FP8 下都明显优于 SiLU；FP8 的确让差距稍大，但额外放大只有约 0.0022 CE。当前证据不支持“收益只来自 FP8 误差”，更符合“主要是激活结构效应，外加较小的精度交互”。后续若要研究 8192 维 Down 输入的长 reduction 是否有特殊影响，应改变 Down 输入维度并补多个 seed。

## 8. SiLU 放在 Attention 哪个位置

这组实验与“共享”不同：矩阵仍然各自独立，只改变 Q/K/V/O 低秩因子之间是否插入 SiLU。

最终 validation macro CE 排序是：

\[
L_{all\text{-}linear}<L_{QK}<L_Q<L_K<L_O<L_V.
\]

其中 all-linear=2.738568，QK=2.739049，差距只有 0.000481。QK 在 28 个验证点中有 23 个优于 all-linear，但最后一点略差；因此只能说 Q/K 联合变换比单独改 Q 或 K 更有希望，尚不能说它稳定超过全线性 baseline。V/O 加 SiLU 的退化更明显。

汇报时不要把这组说成“QK 共享”：这里研究的是 **Q/K 各自低秩路径中的激活位置**，不是让 Q/K 共用 \(A\)。

## 9. SVD 初始化

动机是：线性矩阵的截断 SVD 给出 Frobenius 范数下的最佳 rank-\(r\) 近似，或许比独立随机初始化两个低秩因子更好。

实际比较：

- A：两个低秩因子独立初始化；
- C：先构造随机 dense \(W\)，做 rank-512 截断 SVD，再分别匹配两个因子的初始化 RMS。

30B 结果中，C 相比 A 的最后 1000 step train CE 改善 0.001715；最后共同验证 macro CE 改善 0.003266，10/11 个验证源方向更好。收益方向较稳定，但量级很小，仍需另一个配对 seed 才适合升级为默认初始化。

理论上的限制也很清楚：模型实际计算的是 \(B\phi(Ax)\)，中间有 SiLU 或 RMSNorm，所以 \(BA\) 对 dense \(W\) 的最佳线性逼近，并不等于整个非线性映射的最佳初始化。这也是 SVD 没有带来大幅提升的可能原因。

## 10. 两篇 SVD/稀疏相关工作的定位

### LOST

LOST 将 dense 权重拆成低秩分支与稀疏补偿分支，形式可概括为

\[
Y=\gamma\,\operatorname{SiLU}(XA)B+(1-\gamma)XW_s.
\]

低秩分支来自较大奇异值，稀疏分支保留部分未被低秩近似覆盖的坐标。我们目前的疑问是：稀疏 support 在训练前固定，训练中不重新选择，适应性有限；加入 SiLU 和加权混合后，低秩分支也已经偏离原始 SVD 最佳逼近的线性理论。这目前是文献判断，不是本项目已经完成的训练结果。

### Prompt-aware Dynamic Rank Selection

标准 SVD 优化的是

\[
\min_{W_k}\lVert W-W_k\rVert_F,
\]

但真正关心的可能是校准激活上的输出误差：

\[
\min_{W_k}\lVert WX-W_kX\rVert_F.
\]

若 \(XX^T=SS^T\)，可以转成对 \(WS\) 做低秩近似。问题是 \(S\) 依赖校准数据或 prompt，跨域时可能失配；动态路由又会引入额外参数、显存和执行开销。这条线用于理解 activation-aware SVD，目前不属于主实验结论。

## 11. Down 输出子空间旋转

这组只修改 Down 的最终读出：

- Identity：\(y=Bz\)；
- Static：对所有 token 使用同一个可学习旋转；
- Dynamic：根据 token latent 生成旋转，从训练开始就启用；
- Dynamic residual：以 Identity 精确初始化，再通过可学习门逐步打开动态旋转。

30B 最后 1000 step train CE：

| 变体 | CE |
| --- | ---: |
| Identity | 2.496660 |
| Static | 2.497556 |
| Dynamic | 2.502272 |
| Dynamic residual | **2.495650** |

最后共同 validation macro CE：Identity 2.759772，Static 2.761714，Dynamic 2.762842，Dynamic residual **2.756389**。

结论是：固定旋转没有收益，从随机动态旋转开始训练明显更差；从 Identity 开始逐步开放动态分支有小而一致的改善。验证收益约 0.00338 CE，仍然很小，并且当前实现吞吐更低，所以它只值得补一个 seed 和角度/门值分析，不是主架构突破。

## 12. 其他方向如何放在汇报中

- **SiLU Q/K/V 联合**：若没有完整、匹配的结果表，列为待验证组合，不要和已完成的 QK 激活实验混为一谈。
- **LOST、prompt-aware SVD**：属于文献调研和候选设计，不应说成我们已经复现或证实的结果。

## 13. 建议的五分钟汇报顺序

1. **问题：**大线性层低秩化后，能否让功能相关的投影共享低维表示，并保持模型质量？
2. **公平性：**所有主要比较匹配参数量和训练预算，共享省下的参数用于扩大 rank。
3. **共享结果：**KV 和 Gate/Up 共享有效；QK 较差；QKV 约束过强。
4. **机制结果：**Down 中 Identity≈RMSNorm，二者明显优于 SiLU；主要问题是 SiLU 改变 latent 方向，不是 RMSNorm 带来特殊增益。
5. **数值精度：**BF16 和 FP8 都复现同一趋势，FP8 只小幅放大差距。
6. **补充尝试：**SVD 初始化和 identity-init 动态旋转只有小收益，优先级低于主共享结构。

## 14. 汇报时需要保留的限制

- 多数结构结论目前只有一个训练 seed，曲线上的多个 step 不是独立重复实验。
- 对齐比较必须使用相同 optimizer step/token，不能拿不同进度的最后一点比较。
- Validation macro CE 是 11 个数据源等权平均，不是 BPB，也不是按 token 加权的总 CE。
- 参数量匹配不等于算力和吞吐匹配。
- 当前验证 evaluator 存在已知 label-mask 对齐问题；各组使用同一实现，因此相对差值仍可参考，绝对 CE 需要保留限定。
- 热力图说明表示如何变化，但不能单凭可视化推出最终性能因果。

## 15. 可追溯的完整报告

- Down 激活四组：`scripts/analysis/Down_Identity_Ablations/results_four_runs/report.md`
- FP8/BF16 四组：`scripts/analysis/Precision_Ablations/results_four_runs/report.md`
- Q/K 激活组合：`scripts/analysis/QK_Combination_Analysis/FINAL_RESULTS_20260913.md`
- SVD 初始化：`scripts/analysis/Initialization_Analysis/TRAIN_CE_20260913.md`
- 输出旋转：`scripts/analysis/Down_Output_Subspace_Ablations/RESULTS_20260922.md`
