# Dense 单点 SiLU：五个最终 checkpoint 转 HF

本批次为后续由用户选择的 benchmark 准备 HF 模型，不重训、不启动评测。范围是 Q、K、Up、Gate **单点** SiLU 消融及原生 Dense 对照，不是 Down 激活四格。

## 输入与比较范围

五个实验均取 30B 训练的 `stage1/step14306`，从保存的配置和 DCP 元数据识别实际架构。

| 标签 | W&B / checkpoint run 名称 | 唯一参数量 | HF 子目录 |
| --- | --- | ---: | --- |
| Dense | `olmo3-1b-dense-baseline-bs256-seq8k-30b-wsd-fp8-5e-3` | 1,484,916,736 | `dense-step14306` |
| Q | `olmo3-1b-dense-matched-q-only-silu-r1024-fp8-30b` | 1,484,916,736 | `q-step14306` |
| K | `olmo3-1b-dense-matched-k-only-silu-r1024-fp8-30b` | 1,484,916,736 | `k-step14306` |
| Up | `olmo3-1b-dense-nearmatched-up-only-silu-r1632-fp8-30b` | 1,483,868,160 | `up-step14306` |
| Gate | `olmo3-1b-dense-nearmatched-gate-only-silu-r1632-fp8-30b` | 1,483,868,160 | `gate-step14306` |

Q/K 只将相应的一个 Dense 投影替换为 rank1024 的 `SiLU(XA)B`，其余投影保持 Dense。Up/Gate 只替换相应 FFN 投影，rank1632；它们因向下取整比 Dense 少 1,048,576 参数（约 0.0706%），属于近似匹配，不应称为严格同参数。这里没有使用 C2/C6 的共享低秩 FFN。

同预算、相同训练 token 数的下游比较可检验训练 CE 的差异是否对应任务效果，但单种子的小差异不能自动视为可靠优势。原训练的初始化方式不同，Dense 与消融的 rank microbatch 也分别为 65536 与 32768；这些限制需要保留在最终结果说明中。

输入根目录：

```text
/mnt/shared-storage-gpfs2/intern-pretrain-shared02/taoji/ckpts/Olmo3-MoSE
```

完整源路径为 `<输入根>/<run 名称>/stage1/step14306`。不回退到其他 step，也不按 run 名称推断模型结构。

## 执行方式

所有权重都调用 `olmo_mose.hf_export.exporter.export_checkpoint`。本批次脚本只定义选择范围、参数结构保护和批次状态，不是五套专用转换实现。

- [export_dense_single_silu.py](export_dense_single_silu.py)：先核对五个 checkpoint 的模型键、形状、结构和参数量，再逐个导出；记录每个成功或失败的结果。
- [run_dense_single_silu_agent_8gpu.sh](run_dense_single_silu_agent_8gpu.sh)：在调度节点内安装隔离的 HF 依赖并执行转换，不修改原训练环境。
- [dense_single_silu_job.json](dense_single_silu_job.json)：实际提交记录。

2026-10-08 19:49:46 +08:00 已确认创建任务：

```text
xtuner-mose-hf-single-silu-1008-1949-86104329
agent / ailab-llmagent / llmagent_gpu
1 节点 × 8 GPU，priority=1，32 CPU，262144 MiB
```

转换以一个 CPU 进程读取模型张量并写分片，8 卡是用户规定的申请规模，不表示转换会使用八个 GPU 进程。HF 依赖和读取计算均在调度任务内进行。

输出根目录：

```text
/mnt/shared-storage-gpfs2/intern-pretrain-shared02/shijiayang-p/ckpts/huggingface/dense-single-silu-step14306-20261008
```

开发机对应的共享路径为 `/mnt/shared-storage-user/intern-pretrain-shared02/shijiayang-p/ckpts/huggingface/dense-single-silu-step14306-20261008`。

## 产物与完成标准

2026-10-08 **19:55:22 +08:00 五个模型全部导出完成**；19:55:52 调度器确认 `Succeeded`、active=0、succeeded=1。每个模型为 7 个 safetensors 分片，保留 FP32 master 权重；五个模型的张量数据合计 29,689,946,112 字节（约 29.69 GB，不含 tokenizer 与配置）。完整结果见 [dense_single_silu_exports.json](dense_single_silu_exports.json)。

已从开发机核对五套转换记录、权重分片、配置、独立推理代码与 tokenizer 文件存在且可读。初次 GPU 容器导出目录继承 `mkdtemp` 的 700 权限，已将本批目录修正为 755，并在共享 exporter 的原子发布前设置该权限；原 checkpoint 未改动。

使用 `dtype=auto` 保留 checkpoint 原始张量 dtype；训练启用 FP8 不表示导出权重是 FP8 推理量化产物。分片上限为 1GB，保留 tokenizer、特殊 token ID 和 padded embedding 行。

每个 HF 子目录包含模型配置、独立推理代码、safetensors 分片、tokenizer、generation 配置和 `conversion_manifest.json`。同目录的临时写入全部成功后才改名为正式产物。非空目标拒绝覆盖，原始 checkpoint 不修改。

根目录保存：

- `cohort_status.json`：五条结构审计、源路径、导出状态、输出文件及错误。
- `exports.json`：可由统一 batch 接口使用的源/目标列表。
- `conversion.log`：任务内启动与转换日志。

只有五条均为 `exported` 且批次状态为 `complete` 才表示这批权重全部导出完成。调度器创建成功、排队或仅生成目录都不表示完成。

本次完成脚本检查、实际五个 checkpoint 的结构审计与权重导出，没有执行 forward。按用户要求，数值验证代码已准备，状态保持 `not_run`；benchmark 等待用户选择。后续可通过统一 `validate` 命令检查原生/HF 权重、logits、CE 和解码接口，参见 [统一转换指南](README.md)。
