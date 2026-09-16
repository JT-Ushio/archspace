# Down 内容相关输出子空间消融

本目录是独立的新实验；不覆盖 Down Identity、初始化或 Attention 消融。
只准备代码和脚本，不自动提交。所有旧配置默认 `down_output_rotation=none`，
不增加参数/state_dict key，不改变旧分支运算；旧 checkpoint 仍可严格加载。

## 四组

| 脚本 | Down 读出 | 要回答的问题 |
|---|---|---|
| `run_down_identity_fp8.sh` | `y = Bz, z = Ah` | 同配置基线 |
| `run_down_static_fp8.sh` | `y = R0 Bz` | 静态重参数化/额外优化变量是否已有效 |
| `run_down_dynamic_fp8.sh` | `y = R(z) Bz` | 内容相关方向是否有额外收益 |
| `run_down_dynamic_residual_fp8.sh` | `y + tanh(g)[R(z)y-y]`，`g=0` | 严格从Identity开始，再逐渐启用内容相关旋转 |

三组 Down latent 都是 Identity，保留原来的 down_nonlinear_u/v 名称、矩阵、初始化。
其他 Q/K/V/O/Gate/Up bottleneck 均为 SiLU，FFN 主 SwiGLU 门控仍为 SiLU(gate)*up。
不改 outer FFN RMSNorm、残差、Attention/RoPE、共享开关或优化器。
所有 rank=512，r1=down_r1=0，Gate/Up 不共享，Attention 不共享，无 bias。
不是 all-linear 基线，也不是在旧 Down-RMS 上叠加旋转。

## 算法与精度

默认 d=2048、r=512、G=32，每组64通道/32对相邻通道，共享一个角度。

动态组：
`u = z / sqrt(mean(z^2) + 1e-6)`；静态组：`u = ones(512)`，所有 token 相同。
两组均使用 `W[32,512]`，`a = 0.5*tanh(Wu)`，
`c=(1-a^2)/(1+a^2)`，`s=2a/(1+a^2)`。
每对变换为 `(c*y_even - s*y_odd, s*y_even + c*y_odd)`。
等价角度 `theta=2*atan(a)`，绝对值不超过约53.13度，不计算大旋转矩阵。

静态/动态每层各新增16384参数，16层共262144；基线无新增参数。
静态组虽然同参数数量/形状，但只有32个有效控制自由度，其优化动力学不等价于动态组。
静态旋转在实数函数层面可吸收进 B；在 FP8 和 Muon 参数化下并非逐步数值等价。
因此动态优于静态/基线是重要证据，但不是对“固定子空间为唯一瓶颈”的证明。

控制器以原始二维 Parameter 保存，不是 nn.Linear，因此不参与现有 Float8Linear 替换；
控制器线性计算、RMS、tanh、旋转逐元素运算显式关闭 autocast、使用FP32，再转回 y.dtype。
FSDP 参数混合精度仍按原配置；转换到 FP32 计算不意味着输入/参数未经过 BF16 舍入。
原有 Q/K/V/O/FFN 投影继续原 FP8 配置，不另改 fast_accum 等参数。

原Static/Dynamic的W零初始化，初始a=0、R=I；构造/运行时初始化不消耗额外随机数，
原模型初始权重保持配对。
初始化 hook 在 meta->to_empty/FSDP 路径后再次清零 W，避免未初始化数据。
不使用 “W为零则跳过” 的分支，否则可能切断起点控制器梯度。
W 是内部二维矩阵，沿用现有 Muon 参数分组、LR/weight decay，而非新增独立优化器。

Dynamic-Residual新增每组一个gate：`y_out=y+tanh(g)*(R(z)y-y)`。
控制器W使用`std=1/sqrt(r)`的非零正态初始化，32个gate严格零初始化，因此step 0
与Identity逐元素一致。首个反向中gate有梯度、W梯度为零；gate离开零点后W开始获得梯度，
避免W和gate同时为零造成永久死分支。W初始化使用隔离/克隆的随机数状态，不推进基线模型
的初始化RNG；16层额外增加`16*32=512`个gate参数。

## 训练默认设置

直接复用 `Down_Identity_Ablations/_common.py` 的30B配置，仅设置旋转模式和新W&B组名：

- Global BS256，seq8192，每步2097152 tokens。
- microbatch32768 tokens/卡；2×8时累积4次，4×8时累积2次（纯DP/HSDP）。
- 30B tokens，约14306 step；Muon LR0.005，weight_decay0.033，其余继承原基线。
- 原 WSD：warmup2000 step，最后20% decay；非新的 sqrt 调度。
- FP8开启，每1788 step保存（约12.5%），max_checkpoints=None；启动/最终保存可额外增加。
- 原seed/数据顺序配置、评测设置均继承；validation每500 step。
- 独立run_name：`olmo3-1b-down-output-subspace-{identity|static|dynamic|dynamic-residual}-g32-a05-fp8-30b`。
  W&B group：`down-output-subspace-fp8-30b`。

Shell不重复覆盖Python超参数，允许显式末尾CLI overrides；非法旋转组合在merged config校验失败。
重复跑需更换 RUN_NAME，避免自动恢复/覆盖同一输出；不要让两个活跃任务使用相同保存路径。
正式提交仍遵守远程提交skill：指定SSH网关，conda activate MoSE，arch，默认2×8。
此目录脚本只是被 gpu_run 执行的任务入口，不要在共享网关直接运行它们。

监控程序默认每600秒（10分钟）检查一次状态；运行中每轮最多读取一次各任务关键日志。

## 分析计划

主比较 dynamic vs identity，辅助 dynamic vs static；同 step/tokens 比较 validation CE，
辅以末100/500/1000 step train CE、梯度范数、吞吐、显存峰值。
若预算有限，先跑 identity + dynamic，再决定是否跑静态对照。旧Identity结果仅在代码、
配置、初始化、seed、数据/调度与评测口径均一致时复用，不直接混入不同预算/SVD初始化的结果。

从保存checkpoint、同一批固定tokens做后续只读分析（本轮未实现统计任务）：

1. 逐层/逐组 theta 的token分布、方差、tanh饱和比例，确认不是退化为常量/零角度。
2. `||rotated||/||y||` 和 `cos(rotated,y)`：区分尺度偏差和方向变化；FP精度下允许小误差。
3. 对 B 的列空间做QR得到Q，测原始/旋转后读出的离空间能量：
   `||v - Q(Q^T v)||^2 / (||v||^2 + eps)`。B是PyTorch down_nonlinear_v.weight。
   FP8基线已有量化误差，故须同时报告基线值；静态组应额外用 R0B 的列空间计算，
   不能仅凭离开旧 B 的空间就声称内容适应。
4. token输出矩阵谱/有效rank（结合静态对照），不能把跨token线性张成空间变大说成
   单token局部Jacobian超过512或恢复了encoder丢失的信息。
5. 同checkpoint将动态角度置零/替换成校准集逐组均值，比较CE变化（分布外干预，只是补充证据）。

多个token的输出可离开同一个固定列空间，但控制全部来自512维z，信息瓶颈仍存在。
正交前向不保证梯度范数稳定；新增路径的导数、FP32运算/转换和激活内存都需实测。
Controller约16384 MAC/token/layer，仅相当于原Down两次投影5242880 MAC的0.3125%；
不含归一化、tanh、旋转和反向，不能据此声称端到端开销仅0.3%。

## 验证边界

本地测试入口：`tests/test_output_subspace.py`。包括范数/配对方向、梯度、零初始化、
原矩阵/RNG不变、meta materialization、参数计数、state_dict兼容与roundtrip、
实际三组config差异和CPU AOT编译。正式GPU上仍需FP8+FSDP+compile训练/恢复smoke及profile。
没有用CPU通过替代H200端到端验证，也没有修改此前发现的evaluation mask口径。

2026-09-15 本地结果：新模块9项unittest、旧Down Identity/all-linear回归9项均通过；
4个shell入口逐个通过bash -n，git diff --check通过。
新增测试还验证了小型完整Transformer的meta初始化/反向、原有所有权重的逐项一致性、
控制器被Muon划为matrix参数，以及实际Float8模块转换保留控制器的类型/参数对象。
本地环境未安装pytest，因此依赖pytest的另外两个旧测试文件未运行，未安装或升级环境。
torchao导入提示其C++扩展与本地torch版本不匹配；CPU转换检查通过不排除GPU运行时问题。
