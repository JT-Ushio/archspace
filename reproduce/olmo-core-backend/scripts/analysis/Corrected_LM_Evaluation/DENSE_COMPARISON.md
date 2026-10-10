# Dense与低秩模型：统一最终checkpoint验证

2026-10-07 05:57 HKT。只读评估任务 `mose-eval-dense-r1-p1-1x8-arch-1007-0539-5a2-5fbc0` 已核实Succeeded / active0 / succeeded1，JSON日志SHA256校验通过。

**Dense在Train CE与修正后验证CE上仍领先。低秩797M预算内，Train CE选择C6，验证CE选择C2。**

| 配置 | 参数量 | 末1000步Train CE | 修正后验证macro CE@14306 | 验证CE相对Dense |
| --- | ---: | ---: | ---: | ---: |
| Dense | 1,484,916,736 | 2.357587 | 2.627212 | +0.000000 |
| C0 | 797,050,880 | 2.478978 | 2.738866 | +0.111654 |
| C2 | 797,050,880 | 2.475668 | 2.732681 | +0.105469 |
| C6 | 797,050,880 | 2.474763 | 2.734855 | +0.107643 |

## 评估口径与证据

- Dense与七个低秩配置均实际加载最终step14306 checkpoint；统一8个DP/FSDP ranks、8192长度和32768-token rank microbatch，在同11源验证数据上评分。每个源的有效目标数和旧mask选中数均一致。
- Dense修正后macro CE2.627212026；同forward旧mask2.620586710；修正差值约+0.006625316。Macro为11来源CE等权平均，不是全部token混合micro平均。
- 两种mask均来自同次forward；旧W&B Dense验证为step14000，不能把旧表与本次最终step差值全部归因于mask。
- Dense原始训练microbatch65536，本次独立评估改为32768；没有更改训练配置、权重、优化器或checkpoint。
- 原始Dense的两个精确W&B续训片段完整覆盖14306steps；较新片段覆盖3474个重叠step，与已有Dense全程/末期/最后一步Train CE完全一致。
- 实际W&B保存路径与最终config.json的完整model/init_seed核对，原始recipe也与最终metadata一致；原始调度job ID未知，明确记为null，不声称本轮重新核实了其成功状态。
- 不同模型原始训练硬件、恢复和初始化差异仍需保留；统一评估不消除这些归因限制。所有低秩模型仍为单种子；两预算之间的差距不能作为同参数量消融效果。
- 低秩参数减少约46.3%，质量仍未追平原始Dense。

## 来源

- [Dense原始最终验证JSON](DENSE_FINAL_CHECKPOINT_LM.json)
- [两预算对照JSON](DENSE_COMPARISON_FINAL_CHECKPOINT_LM.json)
- [实际Dense checkpoint元数据](DENSE_CHECKPOINT.json)
- [低秩七配置结果](RESULTS.md)
- [任务及SHA256证据](jobs.json)
