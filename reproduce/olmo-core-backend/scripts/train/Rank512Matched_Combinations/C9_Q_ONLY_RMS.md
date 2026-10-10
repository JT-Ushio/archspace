# C9：独立rank512的Q-only SiLU＋Down RMSNorm

用户于2026-10-09要求补齐Q-only SiLU＋RMS@Down、其他中间变换均为Linear的benchmark对照。现有Q-only rank512的Down为Identity；现有C5有UG共享且UG/Down rank为568/520，不能代替本配置。当前未找到该精确配置的已训练checkpoint，因此从头补训后再评测。

| 项目 | C9 |
| --- | --- |
| Q/K/V/O rank | 全部512 |
| Q中间变换 | SiLU |
| K/V/O中间变换 | Identity / Linear |
| Up、Gate | 独立rank512，中间Identity / Linear |
| Down | rank512，中间无gain RMSNorm |
| 参数共享 | 无 |
| 全模型参数 | 797,050,880，落地前核对实际meta计数 |
| 训练 | 30B，global batch256×8192，Muon LR0.005，WSD warmup2000/末20%decay，FP8 |
| rank microbatch | 32768 tokens |
| 种子 | init12536，data34521；不补重复种子 |
| checkpoint | 每1788步，保留全部，最终step14306 |
| 训练资源 | arch pool，1×8，priority1；不在agent训练 |
| benchmark | 原OLMES olmo3:base_easy，192项；agent 1×8，xtuner-前缀，DP8/TP1，每GPU batch32 |
| HF转换 | 复用统一exporter与现有低秩Attention / MoSE SwiGLU adapter |
| 原生/HF数值验证 | not_run，沿用用户暂不数值验证要求 |

Linear/SiLU/RMS只指低秩两因子之间的变换；OLMo原有QK Norm、RoPE、block归一化和原生SwiGLU门控SiLU保留。

使用行向量，注意力投影为：

\[
q=\operatorname{SiLU}(xA_q)B_q,\quad k=(xA_k)B_k,\quad v=(xA_v)B_v.
\]

Attention聚合结果通过独立Linear低秩O投影。FFN为：

\[
u=(xA_u)B_u,\quad g=(xA_g)B_g,\quad h=\operatorname{SiLU}(g)\odot u,\quad y=\operatorname{RMSNorm}(hA_d)B_d.
\]

主要配对：C9对已有Q-only仅改变Down Identity→RMS；C9对C0比较Q-only与K-only；C9对C5比较独立rank512与共享后秩重分配整体组合。以上使用同一797M预算、30B日程及原OLMES任务hash。单种子小差异不宣称显著；先补训再导出，不通过临时切换已有checkpoint激活来冒充训练对照。

W&B完整名称：`olmo3-1b-r512matched-c9-q-silu-down-rms-fp8-30b`。

配置：[c9_q_silu_down_rms.py](../../../cfps/BS256_Seq8K_30B_WSD_FP8/Rank512Matched_Combinations/c9_q_silu_down_rms.py)。训练入口：[run_c9_q_silu_down_rms.sh](run_c9_q_silu_down_rms.sh)。使用节点本地编译缓存，避免已发生的共享tmp配额问题。训练终态及最终checkpoint保存成功后，才提交转换和评测，使用新输出目录并核对实际保存配置。
