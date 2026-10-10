# 固定参数预算：提高 Attention 秩

本组检验 Attention 的 rank512 是否限制质量：把一部分 FFN 投影预算转给 Q/K/V/O，保持模型总参数为 **797,050,880**，不增加 Dense 预算基线。

| 配置 | Q/K/V/O 秩 | UG 秩 | UG 共享 | Down 秩 |
|---|---:|---:|---|---:|
| A1 | 832 | 336 | 不共享 | 352 |
| A2 | 832 | 320 | 共享输入因子 | 448 |

两组 Q/K 因子之间为 SiLU，V/O 为 Identity；UG 因子之间为 Identity，保留原生 `SiLU(G) ⊙ U`；Down 使用无可学习 gain 的 RMSNorm。Attention 不共享，沿用 OLMo 的 QK norm 与 RoPE。

每层投影预算核算：

- Attention：`4 × 4096 × 832 = 13,631,488`。
- A1 FFN：`10240 × (2 × 336 + 352) = 10,485,760`。
- A2 FFN：`18432 × 320 + 10240 × 448 = 10,485,760`。
- 合计 `24,117,248`，严格等于全部独立 rank512 的投影预算；原生 QK norm 另有4096参数，保持不变。

训练设置同当前组合实验：30B tokens、seq8192、全局 batch256、每卡 microbatch32768 tokens、Muon LR0.005、WSD warmup2000/末20% decay、FP8、初始化seed12536/数据seed34521、每1788步保存且不删历史 checkpoint。8/16/32卡均通过梯度累积保持全局 batch。

主要比较完整14,306步历史与共同末1000步 Train CE，并对照 C6、P0qk 和 F 系列。A1/A2同时改变 Attention 与 FFN 容量，属于预算分配消融；A2相对A1还改变共享和FFN秩，不能把其差异全部归因为共享。运行卡数、初始化布局及中断续训情况单独记录，质量收益不等同于吞吐提升。

启动脚本：`run_a1.sh`、`run_a2.sh`。提交遵循低优priority1，有空卡优先arch16/32，否则pool8；不留下arch排队任务，不重复提交活跃或排队尝试。以实际调度ID和完整日志证明启动，结果尚未产生前不称为新最优。
