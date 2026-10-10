# MoSE 训练、HF转换与OLMES迁移路径清单

清点日期：2026-10-10。依据当前启动脚本、实际转换/评测台账、训练启动记录、环境元数据和本地目录实查整理。包含当前六条训练及此前OLMo实验；GLM部署等其他项目不在本清单范围内。本次没有移动文件、修改启动路径或查询最新GPU任务状态。

**需要保留的主干是：完整MoSE工作树、两个外部源码仓库、两个Python环境、原生checkpoint、已导出的HF模型、训练/验证数据、OLMES离线缓存、评测输出和W&B原始记录。**

## 1. 本地路径与GPU容器路径

本地开发机个人根目录：

```text
U = /mnt/shared-storage-user/intern-pretrain-shared02/shijiayang-p
```

现有GPU容器/提交脚本中的对应个人根目录：

```text
G = /mnt/shared-storage-gpfs2/intern-pretrain-shared02/shijiayang-p
```

后文以U/G表示这两个根目录，全部展开后的路径另见[路径CSV](scripts/analysis/Migration_20261010/paths.csv)和[JSON](scripts/analysis/Migration_20261010/paths.json)。当前开发机可读U，G中的相关项目路径在本机不可见；这是容器挂载布局差异，不能据此认定GPU文件丢失。

旧入口 `/mnt/shared-storage-user/shijiayang-p/archspace-MoSE` 已在本地确认是指向 `U/archspace-MoSE/` 的软链接，不是第二份仓库。

## 2. 主迁移清单与当前可见容量

容量来自本地 `du -sh`，显示值是取整后的二进制单位。未挂载目录没有容量数字。嵌套子目录不应再次加进总量。

| 类别 | 本地/实际路径 | GPU运行路径或说明 | 本地容量 | 保留内容 |
| --- | --- | --- | ---: | --- |
| 完整项目工作树 | `U/archspace-MoSE` | `G/archspace-MoSE` | 约12 GiB | 源码、cfps/cfgs、launchers、统一HF adapter/runtime、测试、skills、报告、分析结果和任务台账 |
| OLMo-core源码 | `U/archspace/OLMo-core-muon-fix` | `G/archspace/OLMo-core-muon-fix` | 约79 MiB | 当前Muon版上游源码、训练/验证mix清单 |
| OLMES源码 | `U/archspace/olmes_official` | `G/archspace/olmes_official` | 约88 MiB | `oe_eval`、suite/task定义和官方评测实现 |
| 训练/转换环境 | `U/conda_envs/olmo` | `G/conda_envs/olmo` | 约13 GiB | Python、PyTorch、torchao、Muon相关依赖、FA2/FA3等 |
| OLMES环境 | `U/conda_envs/olmes` | `G/conda_envs/olmes` | 约15 GiB | Transformers、PyTorch及评测依赖 |
| 原生checkpoint当前根 | `/mnt/shared-storage-gpfs2/intern-pretrain-shared02/taoji/ckpts/Olmo3-MoSE` | 当前训练和统一导出的CKPT_DIR/CKPT_ROOT | 未挂载，未统计 | 每条run的完整stage/step目录、模型、优化器、trainer状态和保存配置 |
| HF模型根 | `U/ckpts/huggingface` | `G/ckpts/huggingface` | 可读部分约76 GiB | 本轮19份HF模型、旧SiTU及导出依赖目录 |
| 训练与LM验证数据 | `/mnt/ailab-llmfudan2/olmo3_data_root` | 数据挂载路径，独立于U/G | 未挂载，未统计 | 本轮6915个训练npy和11个验证npy，详见第4节 |
| OLMES离线数据/Hub缓存 | `U/olmes_cache/huggingface` | `G/olmes_cache/huggingface` | 约10 GiB | `datasets/`、`hub/`，保留blobs和snapshots完整关系 |
| OLMES原始输出 | `U/olmo3_outputs/evals/olmes` | `G/olmo3_outputs/evals/olmes` | 约14 GiB | 原始metrics、逐题预测、配置、任务hash、8个worker日志和汇总 |
| 训练work_dir/W&B | `/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE` | WORK_DIR；离线段在其 `wandb/offline-run-*` 下 | 未挂载，未统计 | 训练配置、原始W&B离线段、数据工作文件及相关日志 |
| 训练HF缓存 | `/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/hf_home` | 当前prepare脚本的HF_HOME | 未挂载，未统计 | 训练/原生评估用tokenizer/Hub缓存；与OLMES缓存不同 |

HF根容量存在一个明确限制：第一次失败的转换留下隐藏临时目录 `rank512-c0-c8-step14306-20261009/.full_linear-step14306.export-uy1vkeh7`，本地无读取权限，`du`报告了该目录的Permission denied。因此76 GiB只代表其余可读部分，不是完整精确总量；此临时目录也不是本轮实际评测使用的正式模型。

项目目录的约12 GiB主要位于 `reproduce/olmo-core-backend/scripts/`，包含大量报告/分析产物，不全是代码。**当前工作树中有大量未提交、未跟踪的新架构和实验文件，仅重新clone或只复制Git已跟踪文件会遗漏本轮代码。**

## 3. 环境及项目外依赖

| 环境 | 已核实关键版本 | 源码依赖/迁移事项 |
| --- | --- | --- |
| `conda_envs/olmo` | Python3.12.13；PyTorch2.10.0+cu128；torchao0.15.0；ai2-olmo-core2.5.0；dion0.1.0；FA2 2.8.2，并安装了FA3 | `__editable__.ai2_olmo_core-2.5.0.pth`硬编码指向 `G/archspace/OLMo-core-muon-fix/src`；该源码必须一起保留。环境有`bin/activate`和`bin/conda-unpack`。 |
| `conda_envs/olmes` | Python3.12.13；PyTorch2.8.0+cu128；Transformers4.57.6；tokenizers0.22.2；safetensors0.8.0 | editable安装的`oe_eval`映射指向 `G/archspace/olmes_official/oe_eval`。本地未发现`bin/activate`或`bin/conda-unpack`；当前启动脚本直接设置PATH/PYTHONPATH并调用`bin/python`。 |

最近L1/R0–R3提交记录中的训练容器镜像是：

```text
registry.h.pjlab.org.cn/ailab/ml-base:22.04-pjlab-20260323
```

CUDA构建版本来自实际环境内的`torch/version.py`，两个环境都是cu128；迁移环境不能仅按Python包名忽略这些编译依赖。全部已安装distribution的名称/版本、editable依赖、环境prefix和工具存在情况保存在[environment_versions.json](scripts/analysis/Migration_20261010/environment_versions.json)。没有执行安装、环境激活或GPU验证。

**统一HF导出还有一个环境外的依赖覆盖目录：**

```text
U/ckpts/huggingface/dense-single-silu-step14306-20261008/.hf-export-deps
```

当前14模型的转换launcher直接复用它，通过PYTHONPATH加载Transformers4.57.6、huggingface-hub0.36.0、tokenizers0.22.2、safetensors0.8.0等。沿用现有导出脚本时需要保留这个隐藏目录，或在迁移后重新准备等价依赖。

本地tokenizer素材在：

```text
U/archspace-MoSE/reproduce/olmo-core-backend/assets/dolma2-tokenizer
```

源码主入口都位于 `U/archspace-MoSE/reproduce/olmo-core-backend/`：训练为 `cfps/`、`cfgs/`、`scripts/train/`，HF转换为 `scripts/convert/` 和 `src/olmo_mose/hf_export/`，OLMES为 `scripts/eval/OLMES_Easy/`。

## 4. 训练与验证数据：可以精确列出使用的文件

当前数据mix为 `OLMo_mix_0625_150Bsample`，实际定义文件：

```text
U/archspace/OLMo-core-muon-fix/src/olmo_core/data/mixes/OLMo-mix-0625-150Bsample.txt
```

按 `TokenizerConfig.dolma2()` 的实际identifier `allenai/dolma2-tokenizer` 展开后，包含 **6915个唯一npy文件、65个source label**，共同父目录为：

```text
/mnt/ailab-llmfudan2/olmo3_data_root/preprocessed/dolma2-0625/v0.1-150b/allenai/dolma2-tokenizer
```

完整文件名已列入[train_data_files.csv](scripts/analysis/Migration_20261010/train_data_files.csv)，含source label、相对路径和原绝对路径。这里是训练mix的全部文件清单，不能把每条实验只训练30B理解为原mix只需要任意30B文件。

LM验证mix为 `v3_small_ppl_validation`，包含 **11个唯一npy文件/11个source label**，共同父目录为：

```text
/mnt/ailab-llmfudan2/olmo3_data_root/eval-data/perplexity/v3_small_dolma2-tokenizer
```

逐文件清单在[validation_data_files.csv](scripts/analysis/Migration_20261010/validation_data_files.csv)。验证mix对tokenizer目录名有单独映射，这里的 `dolma2-tokenizer` 没有训练路径中的 `allenai/` 前缀，不能机械套用同一个目录名。

OLMES benchmark使用另一套HF datasets缓存。当前评测入口设定：

```text
HF_HOME           = G/olmes_cache/huggingface
HF_HUB_CACHE      = G/olmes_cache/huggingface/hub
HF_DATASETS_CACHE = G/olmes_cache/huggingface/datasets
HF_HUB_OFFLINE = HF_DATASETS_OFFLINE = TRANSFORMERS_OFFLINE = 1
```

因此按当前离线配方评测时需要这套缓存。它不是上述训练npy目录，也不是训练prepare脚本中的HF_HOME。

## 5. 原生checkpoint与HF权重

近期原生checkpoint的标准布局：

```text
/mnt/shared-storage-gpfs2/intern-pretrain-shared02/taoji/ckpts/Olmo3-MoSE/<RUN_NAME>/stage1/step<STEP>
```

统一转换读取该step的 `config.json` 和 `model_and_optim/.metadata`/DCP分片。**需要恢复训练时应保留完整step目录及相关stage记录，不能用只含模型权重的HF目录代替优化器、trainer和数据加载恢复状态。** 复制以已完成保存的step目录为单位。

最近一轮六条训练的RUN_NAME：

| 实验 | RUN_NAME；在上面的checkpoint根下各有独立目录 |
| --- | --- |
| C9 | `olmo3-1b-r512matched-c9-q-silu-down-rms-fp8-30b` |
| L1 | `olmo3-1b-ffn-residual-l1-m8192-r512-unshared-linear-down-rms-fp8-30b` |
| R0 | `olmo3-1b-ffn-residual-r0-m2048-r1280-unshared-linear-down-rms-fp8-30b` |
| R1 | `olmo3-1b-ffn-residual-r1-m2048-r1280-unshared-linear-down-rms-fp8-30b` |
| R2 | `olmo3-1b-ffn-residual-r2-m2048-r1280-unshared-linear-down-rms-fp8-30b` |
| R3 | `olmo3-1b-ffn-residual-r3-m2048-r1280-unshared-linear-down-rms-fp8-30b` |

从训练记录和launcher声明整理了86条run的保存路径，见[checkpoint_runs.csv](scripts/analysis/Migration_20261010/checkpoint_runs.csv)。这份表区分“有记录的checkpoint绝对路径”和“按launcher规则展开的路径”；**不是在原生checkpoint盘上枚举出的86个现存目录，也不保证脚本声明的实验均已提交。** 数据盘未挂载，不能给出所有step的现存数量或完整原生checkpoint容量。

本轮实际评测使用的HF目录：

| 模型组 | 路径 | 模型数 | safetensors分片数 | 权重文件容量，不含其他文件 |
| --- | --- | ---: | ---: | ---: |
| CoLA、Full-Linear、RMS@Down、Q-only、K-only、C0–C8 | `U/ckpts/huggingface/rank512-c0-c8-step14306-20261009-r2` | 14 | 56 | 44,635,305,280 bytes，约44.64 GB |
| Dense、Q/K/Up/Gate单点SiLU | `U/ckpts/huggingface/dense-single-silu-step14306-20261008` | 5 | 35 | 29,690,053,960 bytes，约29.69 GB |

这19套正式HF模型的权重文件合计 **74,325,359,240 bytes，约74.33 GB**。14模型每套4个分片，5个Dense预算模型每套7个分片；使用 `dtype=auto` 保留原checkpoint的FP32 master权重，不能因训练为FP8把它们当成FP8量化产物。

HF目录需连同 `config.json`、全部safetensors和index、tokenizer文件、独立runtime/configuration代码、generation配置及 `conversion_manifest.json` 完整保留。只复制safetensors可能无法加载自定义结构。

其他HF产物：

- `U/ckpts/huggingface/rank512-c0-c8-step14306-20261009`：第一次转换留下的CoLA重复正式产物及失败临时目录；本轮14模型实际使用的是带 `-r2` 的完整目录。
- `U/ckpts/huggingface/olmo3-1b-situ-up-step75080`：旧SiTU HF模型。
- `U/olmo3_outputs/huggingface/olmo3-1b-cubit-fp8-5e-3-step75080`：旧Cubit比较模型。

`ckpts/huggingface` 根下可读的21个有config模型目录及逐模型分片容量见[hf_models.csv](scripts/analysis/Migration_20261010/hf_models.csv)，包括19套本轮正式模型、1套旧SiTU和1套第一次转换的CoLA重复产物。

## 6. 评测结果与实验记录

评测原始输出根目录：

```text
U/olmo3_outputs/evals/olmes
```

其中本轮两个正式输出组是：

```text
rank512-c0-c8-step14306-20261009/
dense-single-silu-step14306-20261008-gpu-20261008-2124-r2/
```

另外保留了旧SiTU/Cubit结果、第一次Dense评测尝试和benchmark微基准目录。不要只复制最终均值表：task hash、逐题prediction和完整metrics用于复核与继续分析。

仓库内汇总与台账同样需要保留：

```text
reproduce/olmo-core-backend/scripts/eval/OLMES_Easy/results/
reproduce/olmo-core-backend/scripts/eval/OLMES_Easy/*jobs.json
reproduce/olmo-core-backend/scripts/convert/*cohort*.json
reproduce/olmo-core-backend/scripts/convert/*exports*.json
reproduce/olmo-core-backend/scripts/train/*/jobs.json
reproduce/olmo-core-backend/scripts/train/*/RUN_STATUS.md
reproduce/olmo-core-backend/scripts/analysis/
reproduce/olmo-core-backend/CURRENT_EXPERIMENTS.md
reproduce/olmo-core-backend/PROJECT_OVERVIEW.md
reproduce/olmo-core-backend/EXPERIMENT_SUMMARY_20260907_20261007.md
```

W&B实际work_dir在第2节列出的Fudan路径，关键原始记录为 `wandb/offline-run-*`。仓库内 `reproduce/olmo-core-backend/wandb/` 本地只有少量CLI日志，不是全部训练离线段的替代品。

## 7. 历史路径、平台辅助和可重建缓存

| 路径 | 用途 | 迁移判断 |
| --- | --- | --- |
| `/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/ckpts/Olmo3-MoSE` | 迁往GPFS2之前的checkpoint根 | 完整旧实验归档时核对；本次没有证据证明它与GPFS2根是同一份目录 |
| `U/ckpts/Olmo3-CoLA` | 旧CoLA原生checkpoint | 需要保留旧实验时迁移，不能只搬本轮HF目录 |
| `U/ckpts/olmo3-1b-SiTU-up-5e-3` | 旧SiTU原生checkpoint | 需要保留旧实验时迁移 |
| `U/olmo3_outputs/runs/olmo3-1b-setting2` | 旧OLMo3 stage3运行记录/权重 | 历史归档或恢复该run时需要 |
| `U/olmo3_outputs/dataset-cache/olmo3-stage3` | 旧stage3数据索引和工作缓存 | 恢复旧stage3时核对 |
| `/mnt/shared-storage-user/jitao/envs/MoSE` | legacy训练入口环境 | 当前新训练不使用它；网关仅按环境名激活的`MoSE`实际prefix本次没有查询 |
| `/mnt/shared-storage-user/shijiayang-p/anaconda3/envs/olmo_core` | 旧SiTU专用转换环境 | 当前统一转换不使用它 |
| `G/archspace/OLMo-core/src` | 旧SiTU转换脚本的CORE_SRC | 本地对应目录未见；保留旧专用入口时需核对，当前统一转换用Muon版源码 |
| `/mnt/shared-storage-user/jitao/.local/bin/hceph_start` | 当前launcher的HCeph挂载辅助 | 是平台启动依赖；换平台后需接入新数据挂载，不能当作checkpoint或数据目录 |
| `/mnt/shared-storage-user/jitao/tmp_data` | 旧TMPDIR | 临时缓存，可重建；新C9/残差任务改为节点/tmp |
| `/tmp/mose-ffn-residual.*`、`/tmp/mose-rank512.*` | 训练临时/Inductor/Triton/CUDA缓存 | 不需要迁移 |
| `/tmp/mose-olmes-easy.*`、`/tmp/mose-hf-rank512.*` | 评测动态模块、转换临时文件 | 不需要迁移 |
| `U/hf_home/modules` | 旧HF动态模块缓存 | 可重建，不是OLMES dataset缓存 |

## 8. 迁移后需要调整的引用

1. `scripts/prepare/prepare_olmo_gpfs2.sh`：`GPFS2_USER_ROOT`、`OLMO_ENV`、`MOSE_JT`、`DATA_DIR`、`CKPT_DIR`、`WORK_DIR`和训练`HF_HOME`。
2. 训练/评测环境内的editable路径：olmo的 `.pth`、olmes的 `__editable___ai2_olmes_0_1_0_finder.py` 映射。这些绝对路径不会因把源码目录复制到新位置而自动变更。
3. `scripts/eval/OLMES_Easy/run_easy.sh` 及cohort wrapper：`OLMES_GPFS_ROOT`、`OLMES_REPO`、`OLMES_ENV`、`OLMES_CACHE_ROOT`、模型和结果目录。
4. `scripts/convert/` 的launcher和实际要执行的cohort manifest：`TASK_USER_ROOT`、`CORE_SRC`、`CKPT_ROOT`、`OUTPUT_ROOT`、`DEPS_DIR`及每个源checkpoint绝对路径。其中rank512转换launcher的CKPT_ROOT是脚本中的固定值，不能只改个人根目录就认为已覆盖。
5. 原评测输出的 `scoring_model/`：已实查权重、runtime和tokenizer使用指向旧G目录的绝对软链接。需要重建这个视图或修改链接，原metrics/predictions可以原样归档。
6. 保留HF Hub缓存的完整 `blobs/` 与 `snapshots/`，以及完整datasets缓存。复制缓存后需要在迁移环境核实离线加载；当前清点没有验证新路径。

旧日志和 `conversion_manifest.json` 中的原路径属于来源记录，可原样保留；重新执行的launcher/manifest和有效软链接才需要使用新路径，避免批量替换掉历史来源。

## 9. 机器可读附件与核实边界

- [paths.csv](scripts/analysis/Migration_20261010/paths.csv)：27类路径，含本地路径、运行路径、用途、必需程度、本地是否可见及证据。
- [paths.json](scripts/analysis/Migration_20261010/paths.json)：同上，加数据统计、HF模型与容量记录。
- [checkpoint_runs.csv](scripts/analysis/Migration_20261010/checkpoint_runs.csv)：86条记录/脚本声明的run路径，不是磁盘存在性清单。
- [hf_models.csv](scripts/analysis/Migration_20261010/hf_models.csv)：可读HF模型目录、分片数和实际文件大小。
- [train_data_files.csv](scripts/analysis/Migration_20261010/train_data_files.csv)、[validation_data_files.csv](scripts/analysis/Migration_20261010/validation_data_files.csv)：精确展开后的6915/11个数据文件。
- [environment_versions.json](scripts/analysis/Migration_20261010/environment_versions.json)：两个环境的包版本和editable依赖。
- [absolute_path_references.csv](scripts/analysis/Migration_20261010/absolute_path_references.csv)：4633条静态绝对路径引用及来源文件/行号，供定位后续路径修改。包括历史引用，不等同于4633个必须复制的对象。

核实边界：本地环境元数据、源码、正式HF权重和OLMES输出已实查；原生checkpoint盘、训练数据盘、Fudan W&B/训练HF缓存本机未挂载，只能依据配置和已有运行证据列路径。本次没有统计这些未挂载目录的总容量、验证迁移后forward或改变正在运行的任务。
