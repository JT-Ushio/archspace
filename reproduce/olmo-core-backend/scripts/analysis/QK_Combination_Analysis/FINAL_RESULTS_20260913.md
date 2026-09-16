# Q/K 非线性组合：30B 最终结果与后续判断

更新：2026-09-13，香港时间。QK 训练与已有 L/Q/K checkpoint 分析均已完成；
最后一段 W&B 已同步，以下指标来自完整只读导出，而非控制台四舍五入值。

## 核心结论

**Q+K 同时使用 SiLU，比单独 Q 或单独 K 更有希望，但这次尚未取得超过
all-linear 的最终验证优势。** QK 在 28 个对齐验证点中有 23 个优于 L，
最后一次验证略劣于 L、略优于 Q/K。该结果支持继续研究 Q/K 的联合匹配，
不支持宣称“Q/K 非线性已经稳定优于全线性”，也不支持马上扩展 QK+V/O 大扫参。

最终验证排序（CE 越低越好）：**L < QK < Q < K < O < V**。
这些差异来自每条件单个训练 seed，不能据此声称统计显著或等效。

## 1. 完成状态、配置与比较口径

- 成功任务：`mose-all-linear-qk-silu-r3-2x8-0913-40549445`。
  05:58:43 核实 `Succeeded`，active=0、succeeded=2、failed=0。
- 05:58:08 完成 step14306；05:58:13 最终 checkpoint 保存成功；
  05:58:15 日志报告 `Training complete`，rank0 runner exitCode=0。
  周期 step14304 也于05:58:06确认保存成功。
- 最终保存目录：
  `/mnt/shared-storage-gpfs2/intern-pretrain-shared02/taoji/ckpts/Olmo3-MoSE/olmo3-1b-all-linear-qk-silu-fp8-30b/stage1/step14306`。
  此处是训练保存成功和调度器完成证据，**没有额外做最终 checkpoint 重载完整性测试**。
- 最后离线段：`offline-run-20260913_003525-olmo3-1b-all-linear-qk-silu-fp8-30b`；
  在确认零活跃副本后执行 `conda activate MoSE && wandb_sjy ...`，PID635495，
  随后确认该精确目录的 `Finished syncing`。没有同步尚在写入的活跃训练段。
- 使用原2×8完成；没有取消任务切换4×8，也没有创建并发 checkpoint 写入者。

固定设置：global BS256、seq8192、2,097,152 tokens/step、30B预算、
Muon LR0.005、WSD warmup2000/末20%衰减、FP8、所有低秩维度512、无共享、
microbatch4序列/GPU。周期 checkpoint 间隔1788步，保留进度与最终保存。
实际终点14306步=30,001,856,512 tokens（完整 batch 向上取整）。

L 是所有低秩投影中间激活 Identity；Q/K/V/O 表示仅对应投影改 SiLU；
QK 表示 Q、K 同时改 SiLU。**FFN 主 SwiGLU 门控、QK norm、外层 norm、
attention softmax 均保留**。导出配置逐字段检查：除指定激活、命名/保存路径外相同。
配置一致不等于重启后执行逐位一致；前两次中断与重放已在任务台账记录。

三个时间口径严格分开：

| 口径 | Step | Tokens |
|---|---:|---:|
| 训练终点及最终 checkpoint |14306|30.001857B|
| 最后共同 LM validation |14000|29.360128B|
| 最后共同下游评测 |12500|26.214400B|

LM validation 是 **11 个数据源 CE 的等权宏平均**，单位 nats/token；
不是 token 加权总体 CE，也不是 BPB。训练 loss 使用相同末尾100/500/1000步窗口。
下游 BPB 直接使用日志中的 BPB，不从 CE 换算。

## 2. 最终核心指标

| 变体 | Train CE末100步 | 末500步 | 末1000步 | Validation宏CE | 相对L的CE差 | 优于L的数据源 |
|---|---:|---:|---:|---:|---:|---:|
|L|2.462057|2.470634|2.480808|2.738568|0|—|
|Q|2.437697|2.465468|2.477917|2.739864|+0.001295|5/11|
|K|2.436839|2.464622|2.477119|2.741134|+0.002565|4/11|
|V|2.468794|2.477149|2.487092|2.746239|+0.007671|1/11|
|O|2.467333|2.475728|2.485727|2.745838|+0.007269|0/11|
|**QK**|**2.460390**|**2.468799**|**2.478748**|**2.739049**|**+0.000481**|**7/11**|

QK 最后验证比 Q 低0.000814、比 K 低0.002084；比 L 高0.000481。
QK 末尾训练 loss 比 L 稍低，但没有重现 Q/K 单独变体末100步的较大降幅。
这进一步说明不能把末100步训练 loss 排名直接当作泛化排名。
末100步与末500/1000步差异需保留，不能选最好看的窗口。

## 3. 训练过程：中途改善，末期回到 baseline 附近

下表是 QK 相对于各对照的 validation CE 差（负值更好）：

| Step | Tokens(B) | QK−L | QK−Q | QK−K |
|---|---:|---:|---:|---:|
|2000|4.194|−0.002059|−0.003942|−0.005618|
|3500|7.340|−0.000878|−0.000193|−0.001391|
|7000|14.680|−0.002460|−0.001674|−0.003591|
|10500|22.020|−0.004778|−0.000726|−0.005822|
|11000|23.069|−0.000586|−0.003239|−0.003171|
|11500|24.117|−0.002518|+0.000171|−0.003351|
|12000|25.166|−0.001378|−0.002587|−0.004338|
|12500|26.214|+0.000540|−0.001267|−0.002901|
|13000|27.263|+0.000609|−0.001723|−0.003456|
|13500|28.312|−0.000548|−0.001563|−0.002939|
|14000|29.360|+0.000481|−0.000814|−0.002084|

全部28个点：QK优于L为23/28、优于Q为25/28、优于K为26/28。
这些是同一条轨迹上的相关观测，**不是28次独立实验**。
末期围绕零的小幅波动，不宜解读成已经明确收敛到更好或更坏的架构。

### 2×2 交互效应

令收益 `GQ=L−Q`、`GK=L−K`、`GQK=L−QK`，则

`I = GQK − GQ − GK = Q + K − L − QK`。

最后验证 `I=+0.003380`；28个点中27个为正。
解释：在当前 CE 尺度上，联合变体的表现比“把两个单独改动的收益相加”更好。
但最后 Q/K 单独的收益本身为负，**正交互不等于 QK 已经优于 L**，
更不等于找到了因果机制或具有跨 seed 稳健性的协同。

## 4. 最后验证的全部数据源

| 数据源 | L | Q | K | QK | QK−L |
|---|---:|---:|---:|---:|---:|
|c4_en|3.068381|3.066942|3.069222|3.067704|−0.000677|
|dolma_books|2.984642|2.984478|2.983867|2.985913|+0.001271|
|dolma_common-crawl|3.203704|3.202220|3.203426|3.199238|−0.004467|
|dolma_pes2o|2.226194|2.227071|2.226258|2.227253|+0.001059|
|dolma_reddit|3.378412|3.375073|3.378398|3.375264|−0.003148|
|dolma_stack|1.359173|1.373508|1.376313|1.373138|+0.013965|
|dolma_wiki|2.720521|2.721035|2.719333|2.718228|−0.002293|
|ice|3.043470|3.043517|3.050223|3.049879|+0.006409|
|m2d2_s2orc|3.173884|3.173324|3.175383|3.169805|−0.004079|
|pile|2.315926|2.318117|2.317431|2.315048|−0.000878|
|wikitext_103|2.649943|2.653214|2.652615|2.648072|−0.001871|

QK虽然在7/11源上改善，但 stack 和 ice 的退化抵消了其他改善。
尤其 stack 仍有+0.013965，接近 Q-only 的+0.014334：联合 Q/K 没有消除
这个突出弱点。不能事后移除这些源，再把剩余宏平均称为整体改善。
QK相对Q/K分别有6/11、9/11源改善；相对V/O均为11/11。

## 5. 下游评测：并非单向改善

最后共同评测为 **12500步**。下表准确率以百分数呈现，
MMLU使用日志中的 length-normalized accuracy；均为原版指标，非v2。

| 任务/指标 | L | Q | K | QK |
|---|---:|---:|---:|---:|
|ARC Challenge accuracy (%)|25.171|26.706|26.962|25.853|
|ARC Easy accuracy (%)|24.369|23.022|24.495|24.579|
|Arithmetic accuracy (%)|17.096|17.383|16.714|16.237|
|Coding accuracy (%)|46.146|46.542|45.356|45.751|
|Common knowledge accuracy (%)|81.292|81.292|80.714|79.556|
|Logical reasoning accuracy (%)|84.884|82.200|83.453|85.778|
|Pattern accuracy (%)|61.610|64.419|55.805|63.109|
|String operations accuracy (%)|22.888|23.380|21.903|21.739|
|Copycolors accuracy (%)|7.000|10.000|7.000|7.000|
|MMLU humanities normalized accuracy (%)|23.592|25.484|24.718|25.292|
|MMLU other normalized accuracy (%)|27.606|26.527|24.985|27.236|
|MMLU social sciences normalized accuracy (%)|28.632|26.454|26.942|23.919|
|MMLU STEM normalized accuracy (%)|28.496|28.496|25.381|26.209|
|HumanEval gold BPB ↓|0.514693|0.510284|0.503564|0.517807|
|MBPP gold BPB ↓|0.721434|0.719309|0.720152|0.718688|
|Minerva math gold BPB ↓|0.787221|0.787473|0.785545|0.785441|

这些观测没有一致支持 QK 在下游全面更优；例如 humanities改善但social sciences下降。
不混合准确率、BPB、CE成一个无定义的总分，也不把多种同任务指标当独立证据。
完整CSV包含156个各自对齐的指标（含v2和多语言MBPP），不是156个独立任务。

## 6. 与已有 checkpoint 解释性结果合看

已有[checkpoint分析](CHECKPOINT_RESULTS_20260912.md)只涵盖 L/Q/K，**不含本次QK**：
128条固定序列、15,070个有效抽样预测token、4个checkpoint阶段。
最终 FP8抽样CE为 L2.935900、Q2.955609、K2.956209；不能与上面的完整验证宏CE混算。

Q相对L的额外FP8前向精度惩罚，最终点估计+0.006906，配对序列bootstrap
95%区间[+0.002277,+0.011913]。这使Q路径的数值敏感性值得研究；
但只是这些模型/样本下的前向比较，不能区分量化与累加误差，也不能代表训练seed不确定性。
K相应区间含零，不等于证明无差异。Q在部分层注意力更分散，而非普遍更尖锐。

因此当前较谨慎的工作假说是：**Q/K联合变换可能比单侧变换更适配相似度计算，
但仍存在域依赖的损失及数值敏感性；这种适配尚未转化为最终整体优势。**
正交互是这个假说的线索，不是证明。没有新做QK checkpoint分析，不能推断其精度惩罚已经降低。

## 7. 建议下一步（尚未提交）

1. 优先用已保存QK checkpoint补齐同样的固定样本分析，复用原manifest/token选择：
   比较L/Q/K/QK的CE、BF16/FP8前向敏感性及各层头的attention统计。
   特别检查stack/ice，不仅展示改善域。无需先追加完整训练。
2. 如要确认架构收益，再做配对seed的L与QK复现；若要检验交互效应稳健性，
   则需要配对复现完整L/Q/K/QK四格，而非仅复现QK。
3. 暂缓QK+V/O组合扫参。现有V/O单改结果较弱，QK终点也未明确超过L；
   先弄清“中途收益为何末期缩小”和“代码域为何退化”更有信息量。

当前4×8许可没有被用于自动新开实验。新的分析/训练应先确认范围和资源，
不因有空卡就扩展实验矩阵。

## 8. 图、数据和重现

结果目录：[final-comparison-20260913](results/final-comparison-20260913)。

- [QK相对L/Q/K的验证差值图](results/final-comparison-20260913/trends/qk_validation_delta.svg)：
  最能展示中期收益和后期回到baseline附近；负值更好。
- [验证CE全轨迹](results/final-comparison-20260913/trends/validation_ce.svg)与
  [相对L的收益](results/final-comparison-20260913/trends/validation_gain.svg)。
- [2×2交互图](results/final-comparison-20260913/trends/interaction.svg)：
  正值为超出单独收益之和，不是超越baseline的直接判据。
- [训练CE：不重叠100步块均值](results/final-comparison-20260913/trends/train_100step_blocks.svg)。
- 数据：`trends/headline.csv`、`training_windows.csv`、`validation_history.csv`、
  `validation_final.csv`、`interaction_history.csv`、`downstream_last_matched.csv`、
  `comparison_summary.json`、`config_differences.csv`。

QK原始快照：`results/qk-wandb-final-20260913-0601`，API返回14307行/208种key，
唯一step0–14306；所有非空total-token值逐行核对。1789步之后全部12518行时间戳
落在成功r3段之后，没有手工重编号、删除、平均重复记录。step1788的时间戳是恢复时记录。
同步器显示的history索引2594–15112不是最终API的optimizer step，不能直接作训练横轴。
失败尝试日志/原离线段保留；服务器快照本身不是三次尝试的完整独立档案。

五个对照原始快照：`results/controls-wandb-20260912-0159`。
组合目录仅复制导出数据；`provenance.json`记录每个来源文件及SHA256、续跑审计。
原快照不修改。若API未来出现重复step或token不匹配，脚本直接报错，不能静默修补。

重现（CPU，不需要GPU或W&B写入；输出目录须不存在）：

```bash
python scripts/analysis/QK_Combination_Analysis/final_compare.py \
  scripts/analysis/QK_Combination_Analysis/results/controls-wandb-20260912-0159 \
  scripts/analysis/QK_Combination_Analysis/results/qk-wandb-final-20260913-0601 \
  /path/to/new-comparison-output \
  --restored-step 1788 --attempt-started 2026-09-13T00:32:21+08:00
```

4项对比单元测试通过；5张生成SVG均通过XML解析。任务提交和同步使用仓库
remote-gpu-submit skill约束：arch、已授权SSH查询、终止后同步、无重复活跃任务。

W&B来源：
[L](https://wandb.ai/archspace/MoSE/runs/olmo3-1b-all-lowrank-linear-fp8-30b)、
[Q](https://wandb.ai/archspace/MoSE/runs/olmo3-1b-all-linear-q-only-silu-fp8-30b)、
[K](https://wandb.ai/archspace/MoSE/runs/olmo3-1b-all-linear-k-only-silu-fp8-30b)、
[V](https://wandb.ai/archspace/MoSE/runs/olmo3-1b-all-linear-v-only-silu-fp8-30b)、
[O](https://wandb.ai/archspace/MoSE/runs/olmo3-1b-all-linear-o-only-silu-fp8-30b)、
[QK](https://wandb.ai/archspace/MoSE/runs/olmo3-1b-all-linear-qk-silu-fp8-30b)。
