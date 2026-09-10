# Down checkpoint 起步分析

分析已完成的四组 **30B / FP8 训练 / Muon** checkpoint：S、I、R、SR。
不重新训练、不改 checkpoint、不写 W&B、不加载优化器。不要与早期 7.5B precision runs 混用。

## 提交 GPU 任务

与训练任务使用同一个 prepare / hceph 挂载流程，但启动单个 Python 进程，不用 torchrun：

```bash
gpu_run arch 1 1 mose-down-ckpt-stats /mnt/shared-storage-user/intern-pretrain-shared02/shijiayang-p/archspace-MoSE/reproduce/olmo-core-backend/scripts/train/Down_Checkpoint_Analysis/run_down_checkpoint_analysis.sh
```

只需申请 **1 节点、1 GPU**；H200 足够作为起步硬件选择，尚未测量真实峰值显存。
四个模型依次执行。建议分配至少 8 CPU 核、32 GiB 主机内存；具体峰值取决于 checkpoint 读取和验证集索引构建，未做实机保证。
首次准备验证集可能需要额外 CPU/IO；默认复用训练 WORK_DIR 的缓存。

启动脚本默认路径取自四组已保存的 W&B 配置：

```
/mnt/shared-storage-gpfs2/intern-pretrain-shared02/taoji/ckpts/Olmo3-MoSE/
  olmo3-1b-down-activation-{silu,identity,rms,silu-rms-norm}-fp8-30b/stage1
```

脚本寻找各目录中最大 step，要求四组均为 **14306**，同时存在 `config.json` 和
`model_and_optim/.metadata`。不会在最新目录不完整时悄悄回退。加载时还检查模型全部 key、shape、有限性；
不检查/读取 optimizer shard 的内容。DCP 元数据含 pickle，只使用自己信任的 checkpoint。

可在**任务容器环境**设置下列变量，或修改启动脚本对应默认值。提交端的环境变量是否传入容器取决于 gpu_run，不能假定自动透传。

| 变量 | 默认 / 用途 |
|---|---|
| `CKPT_ROOT` | 上面的公共挂载路径 |
| `S_CKPT`, `I_CKPT`, `R_CKPT`, `SR_CKPT` | 分别指定 run 目录或精确的 step 目录 |
| `EXPECTED_STEP` | 14306；分析中期时同时指定对应四个 checkpoint |
| `NUM_SEQUENCES` | 128；首次 smoke test 可设为 4 |
| `TOKENS_PER_SEQUENCE` | 128；每条抽取最多 128 个有效位置 |
| `ANALYSIS_SEED` | 34521 |
| `OUTPUT_DIR` | 默认 `scripts/analysis/Down_Checkpoint_Analysis/results/down-ckpt-analysis-时间戳` |
| `ANALYSIS_WORK_DIR` | 训练 WORK_DIR；验证集索引、读取缓存 |
| `ANALYSIS_CPU_THREADS` | 8；Torch CPU 计算线程，不等于数据集库内部 IO/索引 worker 上限 |

输出目录存在就报错，不覆盖以前的分析。失败后可选择新的输出目录重跑，不会修改 checkpoint。

## 分析内容与边界

1. 从 checkpoint 的 `config.json` 重建真实架构，核对四组训练配方、模型和数据配置；只允许 Down 激活不同。
2. 使用保存的 LM 验证集定义，仅允许 `DATA_DIR` 更换 mix 挂载根路径。不下载 tokenizer，不拿训练数据替代验证集。
3. 按数据源文件平衡抽取固定样本，序列长度保持 8192；短文档有 padding，因此有效 token 数可能显著少于 1M。
4. 自然前向统一使用 **BF16 权重与 GEMM**，保留原有 norm 内部计算方式，关闭 compile；不启用 Float8 转换。
5. 16 层采集 h/a/z/y/o；同一固定位置在四组中一致，padding 不计入。统计包含上下文中的非 padding token，不按 label_mask 排除未评分上下文。
6. 局部重放使用同一个记录下来的 **a**、同一个 decoder B，分别计算 I/S/R/SR。用 FP32、禁用 TF32，保留真实内外 norm epsilon 和外部 gain。
   这不是重新计算高精度 hA，也不是 FP8 量化或累加误差分解。
7. 使用 `logits_to_keep=1` 避免整段词表 logits 分配。**没有计算 validation loss、BPB 或做单层替换后的全模型评估。**

统计在采样 token 上计算；不缓存完整全层激活。包含 token RMS、峰值/RMS、负值比例与能量、top1% 通道能量、
a/z 中心化协方差谱与有效秩、假想 SiLU 小导数比例，以及重放输出的相对 L2 / 余弦距离。
直方图另外每条样本每个点保留最多 256 个坐标值；量化分位数 `sample_abs_*` 和直方图是这份更小样本的统计。
零向量的余弦未定义，单独计数，不强行记为相同方向。

## 输出

- `report.md`：说明和图表索引，不自动宣称因果机制成立。
- `plots/*.svg`：各层统计曲线、a/z 分布与特征谱、局部重放误差曲线。直接浏览器打开，无 matplotlib 依赖。
- `activations.csv`, `replay.csv`, `spectra.csv`, `silu_sensitivity.csv`：四组合并数据。
- `per_sequence.csv`：逐样本统计，方便后续以文本为单位做配对分析。
- `samples.json`：固定样本 ID、来源、token 哈希、抽样位置及有效长度；不保存原始文本。
- `manifest.json`：checkpoint/config/代码哈希、软硬件环境、完成状态；只有 `status=complete` 才表示全部完成。
- `S/`, `I/`, `R/`, `SR/`：各组 CSV、保存的训练配置、小规模直方图样本。

单独重画（不需要 GPU，不读取训练 checkpoint）：

```bash
python scripts/analysis/Down_Checkpoint_Analysis/plot_results.py /path/to/output
```

已有四组模型权重不同，直接比较自然激活是描述性分析；局部重放固定输入/权重，但仍不能预测重新训练后的收益。
不跨模型对齐某个通道的语义，不把秩更高或 Norm 后尺度接近 1 当作效果更好的证明。

## Token 特征热力图（新增，可选）

旧结果仅有聚合统计/直方图，不能反推出某个 token 的特征。新增启动入口：
`scripts/train/Down_Checkpoint_Analysis/run_down_token_heatmaps.sh`。
复用原分析的四组 checkpoint、128 条验证样本、随机种子和抽样位置；不训练、不改变训练设置。
新任务目录使用 `down-token-heatmaps-时间戳`，不覆盖旧结果。

默认为前 **2 条样本**各保存至多 **64 个已抽样有效 token**，覆盖全部 16 层及 h/a/z/y/o。
这些 token 从原统计的 sampled_positions 中等距选取，位置递增，但**不是连续文本**；不拼接不同序列。
a/z 保存全部 512 通道，h/y/o 默认保存 128 个等距通道；RMS/负值比例/负值能量始终使用全部通道计算。
通过 `HEATMAP_SEQUENCES` / `HEATMAP_TOKENS` / `HEATMAP_CHANNELS` 调整大小。
原分析入口默认 `HEATMAP_SEQUENCES=0`，不开启特征保存，兼容原工作流。
快照为 float32 NPZ（无 pickle），每个样本独立保存，内存只暂存一个样本；默认四模型原始数组约 44 MiB，另有统计开销。
保存 token ID、原序列位置、原始通道编号和样本哈希，不下载 tokenizer、不保存解码文本。

提交须遵循仓库 `mose-remote-gpu-submit` skill。当前偏好申请完整 8 卡节点；该分析仍只使用 cuda:0，
不会因申请 8 卡自动并行。与上文历史单卡示例不同，当前建议命令如下（仅供明确要求提交时执行）：

```bash
ssh -CAXY cpu-develop.jitao+root.ailab-llmagent.ws@h.pjlab.org.cn
# 在 SSH 会话内：
conda activate MoSE && gpu_run arch 1 8 mose-down-token-heatmaps /mnt/shared-storage-gpfs2/intern-pretrain-shared02/shijiayang-p/archspace-MoSE/reproduce/olmo-core-backend/scripts/train/Down_Checkpoint_Analysis/run_down_token_heatmaps.sh
```

GPU 任务只负责保存统计和 `S|I|R|SR/token_features/sequence_*.npz`，不依赖 matplotlib。
完成后在本地、安装了 NumPy/matplotlib 的 CPU 环境绘图（每个输出目录必须不存在）：

```bash
python scripts/analysis/Down_Checkpoint_Analysis/plot_token_heatmaps.py /path/to/new/result \
  --output /path/to/new/figures/sequence0 --sequence 0 --layers 1 4 8 12 16
```

每条样本输出 11 张图（PNG/SVG/PDF）：

- 3 张 token×层热力图：全通道 log10(RMS)、负值比例、负值能量占比；4 模型×5 采集点，按采集点共享色标。
- 4 张 a→z token×通道图：逐 token 除以 RMS、固定 ±4 色标，并记录显示截断比例。
- 4 张 a→z 原始幅值的 signed-log 图：`sign(x)*log10(1+abs(x))`，跨模型/所选层统一色标。

图像不插值、不按激活排序通道、也不假定不同模型/不同层同编号通道语义一致。
归一化视图有意去掉尺度，必须配合原始幅值视图与 RMS 图解释；仅两条文本属于案例展示，不代表总体分布。
脚本核对四模型的 token 位置/ID/样本哈希和通道布局，拒绝未完成任务或不对齐的输入。
零向量的负值能量占比按 0 记录，RMS 图以 1e-12 作绘图下限。
这些是 BF16 自然激活图，不是 FP8 误差或训练梯度统计。

CPU 单元测试：`python -m unittest discover -s scripts/analysis/Down_Checkpoint_Analysis -p 'test_token_heatmaps.py'`。

### 同批前向增加的方向统计

`activations.csv` 额外记录近零坐标比例 `|x| <= 0.01 * token RMS`；对于 a/z，
增加逐 token L2 归一化后的中心化协方差有效秩 `direction_effective_rank` 和所有非零采样 token
之间的平均余弦 `direction_mean_pair_cosine`。方向统计排除零向量，并记录有效 token 数。
它们使用全部 128 条样本的原统计抽样位置，不限于用于热力图的两条文本。
每模型 `direction_spectra.csv` 保存归一化后的协方差谱；原 `effective_rank` 与谱保持不变。
两两余弦通过归一化向量之和计算，不构造 token×token 大矩阵；包含不同序列之间的 token 对。

```bash
python scripts/analysis/Down_Checkpoint_Analysis/plot_direction_stats.py /path/to/new/result \
  --output /path/to/new/direction-figures
```

输出原始/去尺度有效秩、方向相似度与近零比例图，以及逐层等权平均的汇总 CSV。
这些统计用于排查幅值对协方差谱的影响，不能将有效秩高直接解释为质量高。
