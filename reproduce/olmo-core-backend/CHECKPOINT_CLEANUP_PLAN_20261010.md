# Checkpoint清理建议与只读盘点

> 本页是初次盘点建议。用户随后授权仅保留最终step14306，且仅清理Olmo3-MoSE根目录；最新实测清理结果见[最终点清理记录](CHECKPOINT_CLEANUP_FINAL_ONLY_20261010.md)。

2026-10-10 22:02 HKT重新核实：C9、L1、R0–R3六条均Succeeded、active=0，最终step14306保存成功。完整末期Train CE与C9的HF导出/OLMES仍待处理；不能因此删除它们的最终原生checkpoint。初次盘点阶段没有删除或移动任何checkpoint。

## 建议保留规则

| 对象 | 保留 | 清理候选 |
| --- | --- | --- |
| 普通、已核实完成的30B实验 | 完整`step0`和完整`step14306`，以及被分析/转换/评测记录引用的step | 其余未引用的中间step，包括与最终保存仅差2步的`step14304` |
| Dense、C0/C2/C6及G1关键对照 | 在上述基础上保留`step7152`，用于后续中期机制分析 | 其他未引用中间step |
| 仍在运行、状态未知、延长训练或最终checkpoint不可核实的run | 暂时保留全部 | 不放入自动候选 |
| HF模型 | 本轮14套rank512/C0–C8完整`-r2`产物、5套Dense单点模型、需要的历史模型 | 旧重复导出、确认已废弃的失败转换临时目录 |
| 配置、W&B、CE历史、评测metrics/predictions | 保留 | 不以删除这些记录代替清理大体积checkpoint |

例如一个run保存了step0、8个周期点和step14306，只保留step0/最终点会少搬8个目录。实际字节数取决于优化器状态等，不能把目录数量比例当作已核实的空间比例。

**按完整step目录处理，不拆`model_and_optim/`中的模型与优化器分片。** DCP分片混合保存状态，直接删除所谓“optimizer文件”可能破坏checkpoint的加载。保留HF权重也不能代替原生训练恢复状态。

## 已在本机实查的三个手工候选

以下容量是普通文件逻辑大小，不保证与文件系统实际回收空间完全相同。旧run缺少本次可核实的调度终态证据，因此只列为手工候选，没有放入脚本自动候选，也没有删除。

| 候选 | 逻辑大小 | 对应保留对象 | 边界 |
| --- | ---: | --- | --- |
| `ckpts/Olmo3-CoLA/olmo3-1b-cola-r512/stage1/step9000` | 8,037,952,436 bytes，约8.04 GB | 同一个run的`step10000` | 两个目录均有配置、DCP metadata及分片；配置读取权限不足，尚未核实较晚点可加载。未在已整理的静态路径引用中发现对step9000的引用；不等同于证明以后不需要它 |
| `ckpts/olmo3-1b-SiTU-up-5e-3/stage1/step25000` | 13,535,216,389 bytes，约13.54 GB | 同一个run的`step75080`，以及已转换的旧SiTU HF模型 | 配置读取权限不足，尚未核实较晚点可加载；当前旧SiTU转换入口读取step75080。中期25k状态是否需要后续分析仍待确认 |
| `ckpts/huggingface/rank512-c0-c8-step14306-20261009/cola-step14306` | 3,198,189,379 bytes，约3.20 GB | `rank512-c0-c8-step14306-20261009-r2/cola-step14306` | 源checkpoint、config哈希、DCP metadata哈希、参数量、dtype和每个权重分片大小匹配；旧权重文件读取权限不足，payload SHA256未完成，不能称字节级核验通过 |

上述相对路径均位于：

```text
/mnt/shared-storage-user/intern-pretrain-shared02/shijiayang-p
```

两个旧中间点合计约21.57 GB；再计入旧CoLA重复HF目录，合计约24.77 GB。迁移容器中的个人挂载别名是 `/mnt/shared-storage-gpfs2/intern-pretrain-shared02/shijiayang-p`。

旧失败转换的隐藏目录 `.full_linear-step14306.export-uy1vkeh7` 另需在拥有访问权限的容器检查后处理。本机无法读其内容，未估算其容量。**不能把 `.hf-export-deps` 当作转换垃圾一起删：当前统一导出使用它。**

实查记录：[手工候选汇总](scripts/analysis/Checkpoint_Prune_20261010/manual_review_candidates.json)、[旧CoLA原生目录盘点](scripts/analysis/Checkpoint_Prune_20261010/local_cola/checkpoint_audit.json)、[旧SiTU原生目录盘点](scripts/analysis/Checkpoint_Prune_20261010/local_situ/checkpoint_audit.json)、[旧CoLA HF重复候选证据](scripts/analysis/Checkpoint_Prune_20261010/duplicate_cola_review.json)。

## 主checkpoint盘的清理量尚待实盘盘点

主目录：

```text
/mnt/shared-storage-gpfs2/intern-pretrain-shared02/taoji/ckpts/Olmo3-MoSE
```

该盘在当前开发环境未挂载，所以本次没有实际枚举其现有step或统计可释放总量。已准备标准库实现的[只读盘点脚本](scripts/analysis/Checkpoint_Prune_20261010/audit_checkpoints.py)，可在正在迁移的、能访问该盘的容器运行：

```bash
python /mnt/shared-storage-gpfs2/intern-pretrain-shared02/shijiayang-p/archspace-MoSE/reproduce/olmo-core-backend/scripts/analysis/Checkpoint_Prune_20261010/audit_checkpoints.py \
  --root /mnt/shared-storage-gpfs2/intern-pretrain-shared02/taoji/ckpts/Olmo3-MoSE \
  --output /mnt/shared-storage-gpfs2/intern-pretrain-shared02/shijiayang-p/ckpt_cleanup_audit
```

输出为`checkpoint_audit.json`和`checkpoint_audit.csv`，包含每个step的路径、逻辑/allocated大小、保留理由和候选总量。**脚本没有任何删除操作。**

当前配套[完成证据名单](scripts/analysis/Checkpoint_Prune_20261010/completed_runs.json)包含41个从训练台账核实的Succeeded/active=0/最终保存成功的run；它不是所有实验的完整磁盘名单。没有终态证据的run、没有有效最终step布局的run、出现大于原最终step的新checkpoint的run、软链接、扫描错误和不完整中间step均保留。已有静态记录引用的step另受保护。

文件扫描不反序列化权重、不加载模型；只检查保存配置和分片布局，不能替代数值验证。进行实际删除前仍需要复核当前任务/迁移进程和具体候选范围。若目的是减少迁移体积，可以先按保留清单只复制需要的step，原盘清理单独处理。
