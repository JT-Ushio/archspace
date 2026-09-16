# 首轮 validation：只作运行检查与早期观察

两组均在 step500（1,048,576,000 tokens）完成同配置的144-batch评测，随后继续训练。
A日志完成于2026-09-13 14:15:17，C于14:15:20（UTC+8）。
数据来自 rjob rank0 日志的三位小数 CE，不是 W&B 完整精度导出。

| 数据源 | A independent CE | C SVD matched CE | A−C（正值表示C更低） |
|---|---:|---:|---:|
| c4_en-validation | 4.191 | 4.197 | -0.006 |
| dolma_books-validation | 4.278 | 4.271 | 0.007 |
| dolma_common-crawl-validation | 4.292 | 4.298 | -0.006 |
| dolma_pes2o-validation | 3.282 | 3.282 | 0.000 |
| dolma_reddit-validation | 4.359 | 4.377 | -0.018 |
| dolma_stack-validation | 2.653 | 2.655 | -0.002 |
| dolma_wiki-validation | 3.980 | 3.985 | -0.005 |
| ice-validation | 4.276 | 4.298 | -0.022 |
| m2d2_s2orc-validation | 4.302 | 4.264 | 0.038 |
| pile-validation | 3.477 | 3.482 | -0.005 |
| wikitext_103-validation | 4.068 | 4.057 | 0.011 |

等权数据源均值：A≈3.9235，C≈3.9242，差约−0.0007。不是token加权总CE或BPB。
结论仅为：此时两组几乎持平，各数据源方向不同，step0的C优势尚未延续到这一评测点。
仍在2000-step warmup内；一次评测、一个seed和舍入日志不能支持显著优劣结论。
不修改超参、不提前停止、不因起点CE追加raw-SVD全量训练。待30B完成和W&B同步后用完整精度结果替换最终分析依据。

## 第二轮：step1000

2026-09-13 14:21:59(A)/14:21:58(C)，2,097,152,000 tokens。两组随后继续训练，
最新已验证A1044/C1060步。下表同样来自舍入到三位小数的控制台日志。

| 数据源 | A CE | C CE | A−C |
|---|---:|---:|---:|
| c4_en-validation | 3.790 | 3.782 | 0.008 |
| dolma_books-validation | 3.844 | 3.828 | 0.016 |
| dolma_common-crawl-validation | 3.910 | 3.901 | 0.009 |
| dolma_pes2o-validation | 2.894 | 2.886 | 0.008 |
| dolma_reddit-validation | 4.006 | 4.011 | -0.005 |
| dolma_stack-validation | 2.095 | 2.095 | 0.000 |
| dolma_wiki-validation | 3.525 | 3.521 | 0.004 |
| ice-validation | 3.885 | 3.891 | -0.006 |
| m2d2_s2orc-validation | 3.867 | 3.851 | 0.016 |
| pile-validation | 3.056 | 3.053 | 0.003 |
| wikitext_103-validation | 3.586 | 3.564 | 0.022 |

等权均值A≈3.4962/C≈3.4894，C低约0.0068。C在8个来源更低、2个更高、1个舍入后相同。
此时仍在warmup中。与500步几乎持平的结果共同保留，不挑选单个有利时间点作为最终结论。

## 第三轮：step1500

2026-09-13 14:28:39(A)/14:28:34(C)，3,145,728,000 tokens。三位小数日志值如下。

| 数据源 | A CE | C CE | A−C |
|---|---:|---:|---:|
| c4_en-validation | 3.644 | 3.641 | 0.003 |
| dolma_books-validation | 3.678 | 3.669 | 0.009 |
| dolma_common-crawl-validation | 3.766 | 3.766 | 0.000 |
| dolma_pes2o-validation | 2.754 | 2.752 | 0.002 |
| dolma_reddit-validation | 3.893 | 3.892 | 0.001 |
| dolma_stack-validation | 1.917 | 1.913 | 0.004 |
| dolma_wiki-validation | 3.356 | 3.353 | 0.003 |
| ice-validation | 3.702 | 3.788 | -0.086 |
| m2d2_s2orc-validation | 3.705 | 3.689 | 0.016 |
| pile-validation | 2.897 | 2.896 | 0.001 |
| wikitext_103-validation | 3.393 | 3.394 | -0.001 |

等权均值A≈3.3368/C≈3.3412，C反而高约0.0044。ICE差异较明显（C高0.086），
必须保留而不能为获得有利结论剔除；其他来源有更小且方向不同的差异。
三个早期时间点的宏平均差值已经换过方向，不能据此声称SVD稳定优于独立初始化。
训练仍在warmup阶段，保留全部观测并等待完整30B终点。

## Warmup终点：step2000

2026-09-13 14:35:24(A)/14:35:13(C)，4,194,304,000 tokens。两组日志均显示
优化器group0/1/2的LR=0.0050，随后继续训练。三位小数日志CE如下。

| 数据源 | A CE | C CE | A−C |
|---|---:|---:|---:|
| c4_en-validation | 3.562 | 3.561 | 0.001 |
| dolma_books-validation | 3.575 | 3.564 | 0.011 |
| dolma_common-crawl-validation | 3.687 | 3.686 | 0.001 |
| dolma_pes2o-validation | 2.675 | 2.670 | 0.005 |
| dolma_reddit-validation | 3.823 | 3.820 | 0.003 |
| dolma_stack-validation | 1.839 | 1.827 | 0.012 |
| dolma_wiki-validation | 3.259 | 3.261 | -0.002 |
| ice-validation | 3.564 | 3.573 | -0.009 |
| m2d2_s2orc-validation | 3.608 | 3.615 | -0.007 |
| pile-validation | 2.809 | 2.808 | 0.001 |
| wikitext_103-validation | 3.272 | 3.279 | -0.007 |

等权均值A≈3.2430/C≈3.2422，差约0.0008，几乎持平。
完整warmup内各点优势方向反复变化；目前不支持初始化方法有稳定收益的判断。

## 稳定阶段首个评测：step2500

2026-09-13 14:42:04(A)/14:41:47(C)，5,242,880,000 tokens。控制台三位小数CE：

| 数据源 | A CE | C CE | A−C |
|---|---:|---:|---:|
| c4_en-validation | 3.471 | 3.464 | 0.007 |
| dolma_books-validation | 3.465 | 3.454 | 0.011 |
| dolma_common-crawl-validation | 3.593 | 3.592 | 0.001 |
| dolma_pes2o-validation | 2.588 | 2.584 | 0.004 |
| dolma_reddit-validation | 3.742 | 3.742 | 0.000 |
| dolma_stack-validation | 1.734 | 1.726 | 0.008 |
| dolma_wiki-validation | 3.159 | 3.156 | 0.003 |
| ice-validation | 3.470 | 3.479 | -0.009 |
| m2d2_s2orc-validation | 3.530 | 3.525 | 0.005 |
| pile-validation | 2.718 | 2.714 | 0.004 |
| wikitext_103-validation | 3.164 | 3.155 | 0.009 |

等权均值A≈3.1485/C≈3.1446，C低约0.0039。仍是单seed单评测点的小差异，
不与相邻点当作独立重复样本来宣称显著性。两组继续按30B计划训练。

## step3000

2026-09-13 14:48:41(A)/14:48:21(C)，6,291,456,000 tokens。以下仍为舍入控制台CE。

| 数据源 | A CE | C CE | A−C |
|---|---:|---:|---:|
| c4_en-validation | 3.417 | 3.411 | 0.006 |
| dolma_books-validation | 3.398 | 3.384 | 0.014 |
| dolma_common-crawl-validation | 3.543 | 3.537 | 0.006 |
| dolma_pes2o-validation | 2.541 | 2.535 | 0.006 |
| dolma_reddit-validation | 3.696 | 3.689 | 0.007 |
| dolma_stack-validation | 1.676 | 1.667 | 0.009 |
| dolma_wiki-validation | 3.098 | 3.090 | 0.008 |
| ice-validation | 3.417 | 3.413 | 0.004 |
| m2d2_s2orc-validation | 3.487 | 3.478 | 0.009 |
| pile-validation | 2.661 | 2.654 | 0.007 |
| wikitext_103-validation | 3.097 | 3.090 | 0.007 |

此时11个来源的C均低于A，但仍需后续时间点/完整预算及配对seeds才能判断稳定性。

## step3500

2026-09-13 14:55:18(A)/14:54:55(C)，7,340,032,000 tokens。舍入控制台CE：

| 数据源 | A CE | C CE | A−C |
|---|---:|---:|---:|
| c4_en-validation | 3.374 | 3.371 | 0.003 |
| dolma_books-validation | 3.354 | 3.342 | 0.012 |
| dolma_common-crawl-validation | 3.501 | 3.498 | 0.003 |
| dolma_pes2o-validation | 2.503 | 2.498 | 0.005 |
| dolma_reddit-validation | 3.658 | 3.652 | 0.006 |
| dolma_stack-validation | 1.640 | 1.631 | 0.009 |
| dolma_wiki-validation | 3.053 | 3.046 | 0.007 |
| ice-validation | 3.364 | 3.355 | 0.009 |
| m2d2_s2orc-validation | 3.445 | 3.435 | 0.010 |
| pile-validation | 2.621 | 2.614 | 0.007 |
| wikitext_103-validation | 3.044 | 3.037 | 0.007 |

等权均值A≈3.0506/C≈3.0435，C低约0.0071。本次11来源均低于A；
与3000步方向一致，但不能将相邻checkpoint当成独立seed。完整预算仍未完成。
两组随后均在3576步（7,499,415,552 tokens）记录checkpoint保存成功，并继续训练。

## step4000

2026-09-13 15:01:58(A)/15:01:32(C)，8,388,608,000 tokens。舍入控制台CE：

| 数据源 | A CE | C CE | A−C |
|---|---:|---:|---:|
| c4_en-validation | 3.347 | 3.346 | 0.001 |
| dolma_books-validation | 3.322 | 3.323 | -0.001 |
| dolma_common-crawl-validation | 3.474 | 3.473 | 0.001 |
| dolma_pes2o-validation | 2.478 | 2.474 | 0.004 |
| dolma_reddit-validation | 3.628 | 3.627 | 0.001 |
| dolma_stack-validation | 1.612 | 1.610 | 0.002 |
| dolma_wiki-validation | 3.024 | 3.020 | 0.004 |
| ice-validation | 3.345 | 3.350 | -0.005 |
| m2d2_s2orc-validation | 3.433 | 3.430 | 0.003 |
| pile-validation | 2.590 | 2.591 | -0.001 |
| wikitext_103-validation | 3.004 | 3.007 | -0.003 |

等权均值A≈3.0234/C≈3.0228，差距再次缩小。各来源方向不同。
保留这个几乎持平的观测，不只报告3000/3500步的C优势。继续完成固定30B预算。

## step4500

2026-09-13 15:08:36(A)/15:08:05(C)，9,437,184,000 tokens。舍入控制台CE：

| 数据源 | A CE | C CE | A−C |
|---|---:|---:|---:|
| c4_en-validation | 3.324 | 3.323 | 0.001 |
| dolma_books-validation | 3.286 | 3.282 | 0.004 |
| dolma_common-crawl-validation | 3.452 | 3.451 | 0.001 |
| dolma_pes2o-validation | 2.457 | 2.455 | 0.002 |
| dolma_reddit-validation | 3.611 | 3.605 | 0.006 |
| dolma_stack-validation | 1.589 | 1.585 | 0.004 |
| dolma_wiki-validation | 2.999 | 2.993 | 0.006 |
| ice-validation | 3.317 | 3.308 | 0.009 |
| m2d2_s2orc-validation | 3.404 | 3.398 | 0.006 |
| pile-validation | 2.566 | 2.563 | 0.003 |
| wikitext_103-validation | 2.975 | 2.977 | -0.002 |

等权均值A≈2.9982/C≈2.9945。C在10/11来源略低，wikitext_103略高。
这些数值来自三位小数控制台日志，不是完整精度W&B导出，也不代表显著性。
两组评测后均继续训练；仍需完成30B和最终同步比较。

## step5000

2026-09-13 15:15:14(A)/15:14:38(C)，10,485,760,000 tokens。舍入控制台CE：

| 数据源 | A CE | C CE | A−C |
|---|---:|---:|---:|
| c4_en-validation | 3.306 | 3.302 | 0.004 |
| dolma_books-validation | 3.273 | 3.255 | 0.018 |
| dolma_common-crawl-validation | 3.438 | 3.433 | 0.005 |
| dolma_pes2o-validation | 2.442 | 2.437 | 0.005 |
| dolma_reddit-validation | 3.598 | 3.589 | 0.009 |
| dolma_stack-validation | 1.572 | 1.568 | 0.004 |
| dolma_wiki-validation | 2.978 | 2.972 | 0.006 |
| ice-validation | 3.294 | 3.288 | 0.006 |
| m2d2_s2orc-validation | 3.385 | 3.385 | 0.000 |
| pile-validation | 2.550 | 2.545 | 0.005 |
| wikitext_103-validation | 2.956 | 2.951 | 0.005 |

等权均值A≈2.9811/C≈2.9750。C在10个来源略低、1个来源在日志精度下相同。
仍为单个配对seed、三位小数控制台值；不可视作统计显著性或完整预算结论。
两组评测后继续推进，无重启或超参数调整。

## step5500

2026-09-13 15:21:52(A)/15:21:14(C)，11,534,336,000 tokens。舍入控制台CE：

| 数据源 | A CE | C CE | A−C |
|---|---:|---:|---:|
| c4_en-validation | 3.291 | 3.289 | 0.002 |
| dolma_books-validation | 3.249 | 3.247 | 0.002 |
| dolma_common-crawl-validation | 3.421 | 3.419 | 0.002 |
| dolma_pes2o-validation | 2.427 | 2.423 | 0.004 |
| dolma_reddit-validation | 3.584 | 3.577 | 0.007 |
| dolma_stack-validation | 1.561 | 1.554 | 0.007 |
| dolma_wiki-validation | 2.960 | 2.956 | 0.004 |
| ice-validation | 3.294 | 3.311 | -0.017 |
| m2d2_s2orc-validation | 3.369 | 3.356 | 0.013 |
| pile-validation | 2.534 | 2.531 | 0.003 |
| wikitext_103-validation | 2.932 | 2.931 | 0.001 |

等权均值A≈2.9656/C≈2.9631。C在10个来源略低，但ICE高0.017。
保留全部来源；单seed中间结果不支持最终优劣或显著性结论。
两组已完成5364步保存，并在5500步评测后继续训练。

## step6000

2026-09-13 15:28:29(A)/15:27:48(C)，12,582,912,000 tokens。舍入控制台CE：

| 数据源 | A CE | C CE | A−C |
|---|---:|---:|---:|
| c4_en-validation | 3.281 | 3.277 | 0.004 |
| dolma_books-validation | 3.248 | 3.230 | 0.018 |
| dolma_common-crawl-validation | 3.410 | 3.405 | 0.005 |
| dolma_pes2o-validation | 2.418 | 2.414 | 0.004 |
| dolma_reddit-validation | 3.570 | 3.565 | 0.005 |
| dolma_stack-validation | 1.547 | 1.545 | 0.002 |
| dolma_wiki-validation | 2.950 | 2.941 | 0.009 |
| ice-validation | 3.261 | 3.258 | 0.003 |
| m2d2_s2orc-validation | 3.364 | 3.355 | 0.009 |
| pile-validation | 2.523 | 2.519 | 0.004 |
| wikitext_103-validation | 2.925 | 2.918 | 0.007 |

等权均值A≈2.9543/C≈2.9479。本次11个来源均为C略低。
这是单seed的中间验证结果，不是完整30B结论，时间点不作为独立重复实验。
两组评测后继续训练，无重启或配置修改。

## step6500

2026-09-13 15:35:09(A)/15:34:21(C)，13,631,488,000 tokens。舍入控制台CE：

| 数据源 | A CE | C CE | A−C |
|---|---:|---:|---:|
| c4_en-validation | 3.268 | 3.268 | 0.000 |
| dolma_books-validation | 3.229 | 3.216 | 0.013 |
| dolma_common-crawl-validation | 3.400 | 3.400 | 0.000 |
| dolma_pes2o-validation | 2.407 | 2.406 | 0.001 |
| dolma_reddit-validation | 3.559 | 3.560 | -0.001 |
| dolma_stack-validation | 1.541 | 1.538 | 0.003 |
| dolma_wiki-validation | 2.935 | 2.935 | 0.000 |
| ice-validation | 3.262 | 3.255 | 0.007 |
| m2d2_s2orc-validation | 3.342 | 3.349 | -0.007 |
| pile-validation | 2.513 | 2.511 | 0.002 |
| wikitext_103-validation | 2.906 | 2.915 | -0.009 |

等权均值A≈2.9420/C≈2.9412，再次接近持平，来源方向不一致。
保留该点，不只选择SVD优势较大的时间点。控制台三位小数、单seed和未完成预算的限制仍然适用。
两组评测后继续训练，无重启或配置修改。
