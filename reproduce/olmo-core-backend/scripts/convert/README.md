# MoSE / OLMo checkpoint 统一转 HF

本目录为当前 `arch/MoSE-no-kvcache` 分支提供统一转换入口。根据 checkpoint 保存的模型配置与实际模块识别架构，不根据 W&B 名称猜配置，也不为 C2、C6、W1 等每个实验单独写转换脚本。

转换保留低秩因子、中间激活、参数共享、可学习标量和动态旋转。产物使用独立的 `MoSEForCausalLM` HF runtime；复制到其他机器后，推理只需要 PyTorch、Transformers 和 safetensors，不依赖 OLMo-core 或此仓库。非线性低秩结构不能通过简单的矩阵相乘变成等价 Dense 模型。

## 1. 支持的架构与文件分工

| 架构或实验 | 转换模块 | 保留内容 |
| --- | --- | --- |
| 原生 Dense OLMo3 | `architectures/dense.py` | Q/K/V/O、SwiGLU、层归一化顺序、QK Norm、RoPE、滑窗 |
| 独立低秩 Attention / QK、KV、QKV 共享 | `architectures/low_rank_attention.py` | 每个投影的实际 rank、激活、gain、共享输入因子 |
| Dense 单点 Q/K/Up/Gate SiLU | `architectures/dense_single_silu.py` | 只替换的投影，其余投影保持 Dense |
| Dense 预算 Up/Gate/Up+Gate Linear + Down RMS | `architectures/dense_ffn_factorization.py` | 独立的输入因子、原生 SwiGLU 外层 SiLU、无 gain Down RMS；Attention 保持 Dense |
| Dense 预算共享 KV | `architectures/dense_shared_kv.py` | Dense Q/O 与共享 KV 因子 |
| MoSE FFN / C0–C8 / P、F、A、W 组 | `architectures/mose_swiglu.py` | UG 完全/部分/不共享、线性与非线性双分支、宽度、Down rank、激活、bias |
| SiTU / RMS-SiTU / clipping | `architectures/channel_control.py` | control、scope、裁剪与 RMS 尺度，以及 FP32 运算顺序 |
| Static / Dynamic / Dynamic residual Down 旋转 | `architectures/output_rotation.py` | 训练后的 controller、残差门、分组和旋转尺度 |
| G1 可学习 Q/K 激活 | `architectures/adaptive_qk.py` | 训练后的 alpha，不恢复成初始的 0/1 |
| L1 / R0–R3 独立 UG 残差 FFN | `architectures/residual_feed_forward.py` | Up→Down latent残差、2048宽残差开关；原block输出Norm和外层残差保留 |
| SVD 初始化 | `architectures/initialization.py` + 实际 Attention / FFN 模块 | 去掉仅用于初始化的 wrapper，不要求原 SVD 工件；读取训练后权重 |

模块位于 `src/olmo_mose/hf_export/`。

- `registry.py`：按**精确类型**注册模块；新子类未注册时直接报错，禁止降级为 Dense。
- `checkpoint.py`：读取保存的配置、checkpoint 元数据与模型权重。
- `audit.py`：meta 设备检查 HF/原生 state_dict 键名、形状、唯一参数量和共享关系。
- `exporter.py`：统一映射、分片、tokenizer、原子写入和批量结果记录。
- `configuration_mose.py`、`modeling_mose.py`、`runtime_ops.py`：随 HF 权重一起分发的独立推理代码。
- `validation.py`：独立、显式启用的数值验证。

当前转换范围是本分支的 Transformer + LMHead 架构，不包括其他分支的跨层 latent cache、MoE、GatedDeltaNet 或未知 RoPE 实现。已支持原生 pre / reordered post / peri norm block；其他 block 需增加适配器。当前实验 attention dropout 为 0；非零 attention dropout 暂拒绝导出。

## 2. 环境与输入格式

在已有 OLMo-core 训练环境安装可选 HF 依赖：

```bash
cd reproduce/olmo-core-backend
python -m pip install -e '.[hf]'
```

核心实现使用 Transformers 4.57 系列 API；项目依赖限定 `>=4.57,<5`。

标准输入是 OLMo-core DCP checkpoint：

```text
<run>/stage1/step14306/
  config.json
  model_and_optim/
    .metadata
    ...DCP shards...
```

接受明确的 `stepNNN`、`stage1`、run 目录或 `model_and_optim` 目录。输入 run/stage1 时选择最大的数字 step；最新 step 不完整则报错，**不会回退到旧 checkpoint**。

也接受配有 `config.json` 的原生 `model.pt`、`.pth` 或单个 `model.safetensors`。PT 必须是纯模型 state_dict 或仅包含 `model` 的包装，不接受 trainer/optimizer 混合文件。HF 已导出的键名不是原生 checkpoint 输入格式。可以通过 `--config /path/config.json` 指定配置；配置支持完整实验 JSON 的 `model` 字段或独立模型配置 JSON，保留序列化的 `_CLASS_` 信息。

## 3. 转换单个 checkpoint

```bash
python scripts/convert/convert_checkpoint_to_hf.py convert \
  /path/to/run/stage1/step14306 \
  /path/to/hf-model \
  --max-shard-size 2GB
```

安装后也可使用 `olmo-mose-to-hf`，或 `python -m olmo_mose.hf_export`，参数相同。

默认 `--dtype auto` 保留 checkpoint 张量 dtype。需要更小的推理文件时可指定 `--dtype bfloat16`；alpha 与旋转 controller 仍保留 FP32。HF 加载端也为这些参数设置 FP32 保留规则。任何 dtype 转换都可能带来数值差异，验证报告应记录所用 dtype 与容差。

默认从实验 `dataset.tokenizer` 获取 tokenizer 标识及特殊 token ID；可指定 `--tokenizer /local/tokenizer`。`--local-files-only` 禁止下载 tokenizer。没有标识时明确报错，或者显式用 `--skip-tokenizer` 导出仅包含模型的产物；后者不能直接用 AutoTokenizer 加载。Embedding 的 padded vocabulary 不裁剪。

导出流程：

1. 读取配置与 DCP 元数据，meta 构建原生模型和 HF 模型。
2. 严格核对 checkpoint 所有模型键名与形状，以及 HF 参数结构。
3. 逐个读取模型张量并写 safetensors 分片，**不读取 DCP 优化器状态**。
4. 共享参数先检查 checkpoint 中各别名值逐值相等，再保存一份；HF 加载时恢复共享 Parameter 对象。
5. 保存 HF 配置、独立 runtime、generation 配置、tokenizer 和转换记录。
6. 临时目录全部完成后原子改名为目标目录；非空目标拒绝覆盖。

单个大张量可能超过设定分片上限。DCP 的主要内存开销为一个输出分片加一个输入张量，不需要同时物化原生和 HF 两个完整模型；flat PT 使用 mmap。磁盘要求是完整 HF 产物大小加写入中的临时目录，不改动原 checkpoint。

产物包含：`config.json`、`generation_config.json`、三份独立 runtime 文件、safetensors 与可选 index/tokenizer，以及 `conversion_manifest.json`。转换记录包含源配置 hash、DCP metadata hash、step、dtype、共享组、结构审计和数值验证状态。不会复制含有训练回调等信息的完整训练配置。**导出成功不表示数值验证通过**，默认记录 `not_run`。

加载示例：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "/path/to/hf-model", trust_remote_code=True, dtype="auto"
).eval()
tokenizer = AutoTokenizer.from_pretrained("/path/to/hf-model")
inputs = tokenizer("Hello", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=32)
```

使用这里复制的独立 runtime 需要 `trust_remote_code=True`。需要改 dtype 时优先在 `from_pretrained` 指定，保持特殊 FP32 controller 的加载规则。

## 4. 批量转换已有实验

Dense 单点 Q/K/Up/Gate SiLU 与 Dense 对照的最终 step14306 转换批次、参数量、实际任务和产物路径见 [五模型转换记录](DENSE_SINGLE_SILU_EXPORT.md)。

创建 JSON 列表，各条指定 checkpoint 和唯一输出目录；相对路径以 manifest 所在目录为根：

```json
[
  {"checkpoint": "/ckpts/C2/stage1/step14306", "output": "/hf/C2"},
  {"checkpoint": "/ckpts/C6/stage1/step14306", "output": "/hf/C6"},
  {"checkpoint": "/ckpts/W1/stage1/step14306", "output": "/hf/W1"}
]
```

```bash
python scripts/convert/convert_checkpoint_to_hf.py batch exports.json --dtype bfloat16
```

每条使用相同入口与注册表。某条失败后继续处理其余条目，并持续写入 `exports.results.json`；存在失败时进程退出码为 1。排队中、未完成或文件不可见的实验不能视作已转换。批量命令不提交 GPU 任务，也不上传 Hub。

## 5. Forward 参数如何对应

| 原生接口 | HF 接口 | 语义 |
| --- | --- | --- |
| `input_ids [B,T]` | `input_ids [B,T]` | token ID，位置默认从 0 开始 |
| `input_embeddings` | `inputs_embeds` | 跳过 embedding 查表、scale 和 norm；HF 只能与 input_ids 二选一 |
| 无 labels 时返回 Tensor | `.logits` | 对比同一位置、全部 padded vocabulary 的 logits |
| `logits_to_keep` | `logits_to_keep` | 0 保留全部；正整数保留末尾 N 个；Tensor 按索引选取 |
| 原生 labels 已错位 | HF labels 未错位 | HF 计算 logits[:, :-1] 对 labels[:, 1:] 的 CE；-100 忽略 |
| 原生 KV manager | `past_key_values` / `cache_position` | HF 提供普通 DynamicCache 或旧式 tuple，不是训练架构消融 |
| 无标准 padding mask | `attention_mask` / `position_ids` | HF 2D mask 标记有效 key；左 padding 的位置按有效 token 累计 |
| Packed document / CP 参数 | 无直接对应 | `doc_lens`、`cu_doc_lens`、CP 切片和外传 RoPE buffer 不静默接受 |
| 原生 z loss / 自定义 loss divisor | 无直接对应 | 此 HF runtime 的 labels loss 是普通因果 CE |

`attention_mask` 支持 2D key mask，以及 4D bool/additive mask，并叠加原生 causal/sliding-window 限制。已存历史 token 的 mask 需要一并传入。当前实现不支持 StaticCache、滑窗截断 cache 或模型并行；cache 保留完整标准 K/V 历史，逐层 attention 仍遵循原生窗口。

推理计算按原生顺序实现：投影与中间激活 → clip → 完整 QK norm 或 head norm → 原生 half-split RoPE → attention → output gate / output projection。Reordered OLMo3 block 使用 Attention/FFN **输出之后**的 RMSNorm。RMSNorm、QwenRMSNorm 的 dtype 回转顺序不同，导出分别保留；Down 无 gain RMS 不会添加新 gain 参数。

## 6. 先做结构检查，数值验证以后显式跑

```bash
# 读取 config 与 checkpoint 元数据，不读 DCP 张量、不执行 forward。
python scripts/convert/convert_checkpoint_to_hf.py inspect /ckpts/C6/stage1/step14306 \
  --report c6-structure.json

# 只有保存的 config.json 也可以检查。
python scripts/convert/convert_checkpoint_to_hf.py audit-config config.json --report structure.json

# 检查仓库每个 CFP 的序列化配置与 HF 参数结构；仅 meta 构建。
python scripts/convert/audit_experiment_configs.py --report configs-structure.json
```

审计报告记录 native/HF forward 签名、模块类别、参数数和输入差异。它证明参数结构相容，不能证明完整算子数值等价。

后续获得实际 checkpoint 并明确需要验证时：

```bash
python scripts/convert/convert_checkpoint_to_hf.py validate \
  /ckpts/C6/stage1/step14306 /hf/C6 \
  --device cpu --dtype float32 --atol 1e-5 --rtol 1e-4

# BF16 可以从较宽容差起步；检查最大/平均误差与逐层误差后再决定是否可接受。
python scripts/convert/convert_checkpoint_to_hf.py validate \
  /ckpts/C6/stage1/step14306 /hf/C6 \
  --device cuda --dtype bfloat16 --atol 0.03 --rtol 0.02 \
  --input-ids test-input-ids.json
```

token ID 文件为 `[T]` 或 `[B,T]` JSON。默认短序列随机 token 验证可执行性；质量敏感验证应使用真实文本 token、多个长度，包含跨过滑窗边界的输入。大模型验证需同时容纳两个模型；`--no-intermediates` 可减少保存中间输出的 CPU 内存。

验证代码提供：

- AutoModelForCausalLM 从产物加载，检查 missing/unexpected/mismatched keys 与共享 Parameter。
- 每个 reloaded 权重对原生参考逐值、dtype 检查。
- 同一未 padding 输入的 logits，以及每层 block、Attention、FFN、Q/K/V/O 中间输出。
- 同一错位 target 上的 CE，HF labels loss、inputs_embeds、logits_to_keep。
- 逐 token cached decode、prefill 后 decode、左 padding 与带 padding 的缓存 decode。
- `.generate()` 的基本执行与长度。

参考是 checkpoint master 权重的 eager、非 FP8 原生推理。这个检查不能保证复现训练时 Float8 GEMM 的位级结果，也不替代完整 benchmark。验证失败保存误差报告并返回非零；通过才将 report 标记为 passed。

`tests/test_hf_export.py` 包含小模型的架构矩阵、control/scope、DCP/flat 转换、分片 reload、无项目导入加载、批量失败与不完整 checkpoint 用例：

```bash
python -m pytest tests/test_hf_export.py -q
```

**根据本次要求，数值验证代码已写好但没有运行。** Dense 与四个单点 SiLU 的 step14306 已导出，完成记录见 [五模型转换记录](DENSE_SINGLE_SILU_EXPORT.md)；随后14个 rank512/C0–C8 模型也已导出，记录见 [14模型转换清单](rank512_c0_c8_exports_r2.json)。其他批次需分别记录导出进度。配置扫描、权重导出及benchmark完成均不能描述为原生/HF forward 数值验证通过。

2026-10-08 对当前 `cfps/` 的 89 个实验入口完成了序列化配置与 meta 参数结构核对，89 个通过；摘要在 [config_structure_audit.json](config_structure_audit.json)。本次另完成 Python 编译与 Ruff 检查，未执行上述数值测试。

## 7. 新架构的扩展约定

每增加架构或改变已有 forward 计算，必须同时维护对应 HF 模块：

1. 在 `hf_export/architectures/` 为新类型注册明确适配器，并在 `architectures/__init__.py` 引入。
2. 描述实际已构建模块的 rank、激活、norm eps/gain、共享、RoPE/scale 和新增参数；读取训练后的参数值。
3. 推理算子变化时更新独立 `runtime_ops.py`，禁止依赖训练库，禁止忽略新状态或用 Dense 替代。
4. 使用统一原生→HF 键名映射。特殊映射需求集中维护，所有持久状态必须被结构审计覆盖。
5. 增加小模型 forward、序列 decode 和 checkpoint→HF reload 用例；结构检查与数值检查分别记录。
6. 更新本文支持表。仅增加 rank、宽度、激活等已有配置组合时复用模块，不新增每实验转换脚本。

对应约束已写入项目 `mose-hf-export` skill，并从 `mose-experiment-defaults` 链接。用户要求暂不验证时可以先提交代码和验证入口，但必须明确记录数值状态 `not_run`。

HF 自定义模型分发机制参考 [Transformers 官方文档](https://huggingface.co/docs/transformers/custom_models)。
