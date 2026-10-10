# MoSE 消融实验：各组最优配置

更新：2026-09-30。本版主指标改为完整 30B 训练最后 1000 个 optimizer step
（step 13307--14306）的 Train CE 算术平均，越低越好。数值从离线 W&B history
逐 step 读取，不是控制台每 10 step 的抽样。最终 LM validation 的 11 个数据源
CE 等权宏平均保留为泛化参照。同一 sweep 内的排名可直接比较；跨 sweep 的配置
背景不同，不能把各组赢家直接拼接成一个已经验证过的组合。

## 总表

| 消融组 | Train CE 最优配置 | 末 1000-step Train CE | Validation 最优配置 | 判断 |
| --- | --- | ---: | --- | --- |
| Down 中间激活 | Down RMSNorm，无 learnable gain | **2.496314** | Down RMSNorm：2.756267 | RMS 与 Identity 基本持平；主要收益来自去掉 SiLU |
| O/Down RMSNorm 位置 | O SiLU + Down RMSNorm | **2.496082** | O RMSNorm + Down RMSNorm：2.756907 | Down RMS 稳定；O 用 SiLU 还是 RMS 取决于训练/验证口径 |
| Attention SiLU 位置 | K-only SiLU | **2.477119** | all-linear：2.738568 | K/Q/QK 训练略优，最终验证没有超过 all-linear |
| 同低秩参数预算共享 | KV 共享，K/V rank 680 | **2.488552** | KV 共享：2.747219 | KV 与 Gate/Up 共享在两种口径下都并列领先 |
| Dense 参数量匹配 | 原生 Dense | **2.357587** | 原生 Dense：2.626223 | 高 rank 下 F1 共享没有质量收益 |

## 1. Down 激活四格

共同背景：Q/K/V/O、Gate/Up 的低秩中间激活均为 SiLU；只改变 Down
两因子之间的变换。所有低秩 rank 均为 512，无共享，FP8，30B tokens。

| 代号 | Down 变换 | 末 1000-step Train CE | Validation macro CE |
| --- | --- | ---: | ---: |
| R | RMSNorm(a) | **2.496314** | **2.756267** |
| I | a | 2.496371 | 2.756521 |
| S | SiLU(a) | 2.515614 | 2.776709 |
| SR | RMSNorm(SiLU(a)) | 2.516755 | 2.778165 |

结论不是“RMSNorm 提供了约 0.02 的收益”，而是“去掉 SiLU 提供了几乎全部
收益”。R 与 I 的 Train CE 只差 0.000057，validation 差 0.000254；单 seed 下
不足以声称 RMS 本身稳定优于 Identity。若工程上必须选一个，RMSNorm 是两种
口径的数值第一，并且 7.5B 的 BF16/FP8
四格中都稳定优于 SiLU：最后验证 BF16 改善 0.024302，FP8 改善 0.026494。
这也说明收益不是只由 FP8 引起。

详细结果见
[Down 四格报告](Down_Identity_Ablations/results_four_runs/report.md)和
[精度四格报告](Precision_Ablations/results_four_runs/report.md)。

## 2. O 与 Down 的 RMSNorm 位置

这组保留 Q/K/V、Gate/Up 的 SiLU，仅改变 O 与 Down 的低秩中间变换。

| O | Down | 末 1000-step Train CE | Validation macro CE |
| --- | --- | ---: | ---: |
| SiLU | RMSNorm | **2.496082** | 2.759396 |
| RMSNorm | RMSNorm | 2.498739 | **2.756907** |
| RMSNorm | SiLU | 2.516751 | 2.776099 |
| SiLU | SiLU | 2.517768 | 2.781310 |

Train CE 下，只改 Down 为 RMSNorm 比双 SiLU 改善 0.021686；再把 O 换成
RMSNorm 反而回退 0.002657。Validation 则相反：双 RMS 比 O-SiLU/Down-RMS
好 0.002489。两种口径都确认 Down RMS 是主要收益，但 O RMS 只有很小且方向
不一致的影响。这组也没有在完全相同背景下直接比较 O Identity 与 O RMSNorm，
因此不能断言 O 上 RMSNorm 一定优于 Identity。

## 3. SiLU 位置消融

L 保留原生 `SiLU(gate) * up`、attention softmax、QK norm 和外层 RMSNorm，
但所有新增低秩因子之间均为 Identity。Q/K/V/O/QK 只在名字对应的 Attention
低秩投影中加入 SiLU，其他低秩位置保持 Identity。

| 配置 | 末 1000-step Train CE | 相对 Train 最优 | Validation macro CE |
| --- | ---: | ---: | ---: |
| K-only SiLU | **2.477119** | 0 | 2.741134 |
| Q-only SiLU | 2.477917 | +0.000798 | 2.739864 |
| Q+K SiLU | 2.478748 | +0.001629 | 2.739049 |
| L：all-linear | 2.480808 | +0.003689 | **2.738568** |
| O-only SiLU | 2.485727 | +0.008608 | 2.745838 |
| V-only SiLU | 2.487092 | +0.009973 | 2.746239 |

Train CE 排序为 `K < Q < QK < L < O < V`，最终验证排序为
`L < QK < Q < K < O < V`。QK 在 28 个验证点中的
23 个优于 L，但最后验证仍高 0.000481，所以只能说 Q/K 联合 SiLU 接近
all-linear，不能说已经超过。Q/K 单独加入 SiLU 对训练 CE 略有帮助，却没有
转化为最终验证收益；V/O 的退化更明确。

因此若只按 Train CE，选 K-only；若按最终 validation，选全线性。K/Q/QK/L
之间的差都较小，V/O SiLU 在两种口径下都更差。详细结果见
[SiLU 位置最终报告](QK_Combination_Analysis/FINAL_RESULTS_20260913.md)。

## 4. 同参数预算共享

共同背景：Q/K/V/O 与 Gate/Up 使用 SiLU，Down 使用无 learnable gain 的
RMSNorm。共享节省的参数通过增大共享 latent rank 补回，使各变体与 rank-512
不共享基线处在同一低秩参数预算附近。

| 配置 | 共享与 rank | 末 1000-step Train CE | Validation macro CE |
| --- | --- | ---: | ---: |
| KV | K/V 共享，K=V=680 | **2.488552** | **2.747219** |
| UG | Gate/Up 共享，rank=568 | 2.489119 | 2.747328 |
| UG+QK | Gate/Up rank=568；Q/K rank=680 | 2.493863 | 2.752466 |
| 不共享基线 | 全部 rank=512 | 2.496082 | 2.759396 |
| QK | Q/K 共享，Q=K=680 | 2.496425 | 2.755726 |
| QKV | Q/K/V 共享，rank=768 | 2.500876 | 2.760379 |

KV 与 UG 的 Train CE 只差 0.000567，validation 只差 0.000109，单 seed 下
应视为并列第一；若必须给一个默认值，选 KV。QKV 在两种口径下都最差，说明
把共享范围继续扩大并不能靠提高 rank 补偿。中间三项的细微排序会随训练/验证
口径变化，不应把 0.001--0.004 的差解释成稳定显著性；稳健层级是
`KV ≈ UG` 最好、`QKV` 最差。

## 5. Dense 参数量匹配

这一组与上一组不能混为同一个 sweep。F0/F1 把低秩宽度提高到接近原生 Dense
总参数量；Down 均为 RMSNorm，Attention 为 Identity，Gate/Up 为 SiLU。

| 配置 | 关键结构 | 末 1000-step Train CE | Validation macro CE |
| --- | --- | ---: | ---: |
| Dense | 原生 Dense 投影 | **2.357587** | **2.626223** |
| F0 | 不共享；Q/K/V/O 1024，Gate/Up 1640，Down 1632 | 2.392441 | 2.655574 |
| F1 | KV 共享 1360，Gate/Up 共享 1824，Q/O 1024，Down 1632 | 2.397395 | 2.659801 |

Dense 在 Train CE 和 validation 都最好；在两个低秩模型内部，F0 比 F1 的
Train CE 好 0.004954，validation 好 0.004228。F1 的每卡吞吐比
F0 高约 10.5%，但这是质量换速度，不是共享带来的质量收益。因此低 rank
参数匹配 sweep 中的 `KV ≈ UG` 优势不能外推成“高 rank 时把两者同时共享也
会更好”。详细结果见
[Dense 参数匹配报告](../train/DenseMatched_MoSE/README.md)。

## 推荐表述

当前证据支持以下表述：

1. Down 中间 SiLU 明确有害；Identity 与 RMSNorm 基本等价，RMSNorm 两种
   口径都数值略优。
2. Attention SiLU 若只按 Train CE，K-only 数值最好；若按最终 validation，
   all-linear 最好。V/O SiLU 两种口径都更差。O RMS 与 O SiLU 的差很小且在
   Train/validation 间反向，尚无同背景的 O RMS/Identity 直接比较。
3. 同低秩参数预算下，KV 共享与 Gate/Up 共享并列最好；扩大为 QKV 共享最差。
4. 接近 Dense 参数量时，现有共享组合不如不共享低秩，更不如原生 Dense。

尚未训练过 `all-linear + KV shared`、`all-linear + Gate/Up shared` 或
`O RMSNorm + all-linear elsewhere` 的完整组合，因此这些组件不能直接拼成
“全局最优配置”。
