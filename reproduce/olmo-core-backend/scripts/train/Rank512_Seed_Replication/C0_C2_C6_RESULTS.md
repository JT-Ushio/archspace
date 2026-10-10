# C0 / C2 / C6：两组初始化种子的完整Train CE与最终验证

2026-10-07 12:36 HKT。六次训练均确认Succeeded、active0、完整30B/14306steps及最终checkpoint保存，精确W&B上传均已确认。完整Train CE与新增两次C2最终验证合并在pool 1×8低优任务 `mose-seed-confirm-r1-p1-1x8-pool-1007-1228-6-a9b14` 中执行，任务确认Succeeded、active0/succeeded1，三份输出分别经SHA256校验；C0/C6四次最终验证来自此前已验证的任务。

**两个新增初始化种子均支持：Train CE以C6较低，修正后最终验证CE以C2较低；C2和C6在两个指标上均优于C0。** C6相对C2的Train CE配对平均改善仅0.000423，验证macro CE则高0.001304，不能把训练排名直接等同于整体质量排名。

## 配置与口径

- C0：独立Q/K/V/O及UG rank512；K中间SiLU，其他Attention和UG中间Identity，Down512无gain RMSNorm。
- C2：独立Q512 Identity、K512 SiLU、V512/O496 Identity；UG共享568、中间Identity，Down520无gain RMSNorm。
- C6：除Q中间使用SiLU外，模型配置与C2相同。三组都保留原生SwiGLU的外层SiLU门控。
- 全部797,050,880参数，初始化seed12537/12538，固定data seed34521，pool 1×8低优训练；seq8192、global batch256、Muon0.005、WSD2000warmup/末20%decay、FP8、完整30B。
- 六组完整steps1–14306，无缺失、重复、非正或非有限Train CE；共同排除集合为空，末1000步窗口13307–14306含1000点。
- 均值只包含两个新增初始化种子，不混入原始seed12536。不同模型按相同初始化seed配对，数据顺序固定，不宣称统计显著。

## 两种子均值

| 配置 | 全程平均Train CE | 末1000步平均Train CE | 修正后最终验证macro CE |
| --- | ---: | ---: | ---: |
| C0 | 2.793230025 | 2.479431683 | 2.740639253 |
| C2 | 2.791133491 | 2.475767208 | 2.733279575 |
| C6 | 2.790045210 | 2.475344465 | 2.734584061 |

按用户指定的主要Train CE指标，当前仍选择C6；验证集指标保留C2为优选。共享与预算重分配组合的主要改善已由C2相对C0体现，额外Q SiLU带来更小的训练改善，但未带来验证改善。C2相对C0同时改变共享方式和秩分配，不能将全部收益单独归因于共享。

## 六次运行

| 初始化seed | 配置 | 全程平均Train CE | 末1000步平均Train CE | 最后一条原始Train CE | 最终验证macro CE |
| --- | --- | ---: | ---: | ---: | ---: |
| 12537 | C0 | 2.793364229 | 2.479495967 | 2.509505987 | 2.742605827 |
| 12537 | C2 | 2.790961065 | 2.475723318 | 2.505619049 | 2.733086066 |
| 12537 | C6 | 2.790239032 | 2.475591339 | 2.505630732 | 2.734566862 |
| 12538 | C0 | 2.793095821 | 2.479367399 | 2.508933544 | 2.738672679 |
| 12538 | C2 | 2.791305917 | 2.475811098 | 2.505337000 | 2.733473084 |
| 12538 | C6 | 2.789851389 | 2.475097591 | 2.505316973 | 2.734601259 |

## 配对差值

| 对比（后者−前者） | 12537 Train CE差 | 12538 Train CE差 | 平均Train CE差 | 平均最终验证CE差 |
| --- | ---: | ---: | ---: | ---: |
| C2−C0 | -0.003772649 | -0.003556301 | -0.003664475 | -0.007359678 |
| C6−C2 | -0.000131979 | -0.000713507 | -0.000422743 | +0.001304486 |
| C6−C0 | -0.003904628 | -0.004269808 | -0.004087218 | -0.006055193 |

C6−C2的验证差在12537/12538分别为+0.001480796/+0.001128175，两个新增种子都重复了原始运行的训练与验证取舍。这里报告方向和实际差值，不把两个固定数据顺序的初始化重复解释为统计显著性，也不单独作过拟合的因果归因。

## 最终验证的可比性

六个实际step14306 checkpoint均按有效目标标签mask重评，同11个验证源等权macro；每源内部按有效token平均。每源的正确mask及旧mask整数计数在六组中完全一致，总有效目标5,874,543，同forward旧mask选择5,893,039。C0/C6与C2分两批8卡H200分配执行，不声称六个模型在同一物理节点评估。所有checkpoint的完整模型配置、初始化、参数量和源任务都已核对，评估未训练或保存源模型。

旧W&B的validation_macro_ce来自step14000及旧mask，不与本表修正后的最终step14306验证混用。Dense为1,484,916,736参数，属于另一预算；本报告不宣称已追平Dense。

## 数据与证据

![C0、C2与C6的两种子训练和验证配对结果](../../analysis/Corrected_LM_Evaluation/figures/paired_seed_c0_c2_c6.png)

[SVG](../../analysis/Corrected_LM_Evaluation/figures/paired_seed_c0_c2_c6.svg) / [PDF](../../analysis/Corrected_LM_Evaluation/figures/paired_seed_c0_c2_c6.pdf) / [CSV](../../analysis/Corrected_LM_Evaluation/figures/paired_seed_c0_c2_c6.csv)。[作图脚本](../../analysis/Corrected_LM_Evaluation/plot_paired_seed_c0_c2_c6.py)校验完整训练与六份验证payload的SHA256、源任务、初始化和预算；原有四次C0/C6图保留。图中曲线连接同初始化种子的三个模型，并非训练过程曲线。

- [完整六次Train CE](ALL_SEED_FULL_TRAIN_CE.json)
- [训练与验证配对汇总](../../analysis/Corrected_LM_Evaluation/SEED_C0_C2_C6_COMPARISON.json)
- [六次训练任务与完成证明](jobs.json)
- [冻结Train CE源清单](../UG_Partial_Sharing/completed_seed_segments.json)
- [两次C2精确checkpoint清单](../../analysis/Corrected_LM_Evaluation/c2_seed_cases.json)
- [分析任务及SHA256](../../analysis/Corrected_LM_Evaluation/jobs.json)
- [分析完成日志](../../analysis/Corrected_LM_Evaluation/all_seed_train_c2_final_observation.log)
- [保留的C0/C6四次报告](C0_C6_RESULTS.md)
- [项目概况](../../../PROJECT_OVERVIEW.md)
