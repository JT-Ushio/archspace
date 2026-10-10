# 固定预算下把 UG 参数分配给 Down

这组补充实验检验一个不同于共享比例的问题：所有投影都使用rank512，
是否把过多预算给了UG、而限制了Down？只移动FFN内部的参数预算。

| 配置 | UG rank | UG共享 | Down rank | FFN参数/层 |
|---|---:|---|---:|---:|
| 参照P0 | 512 | 否 | 512 | 15,728,640 |
| F1 | 384 | 否 | 768 | 15,728,640 |
| F2 | 256 | 否 | 1024 | 15,728,640 |
| F3 | 480 | 是 | 672 | 15,728,640 |
| F4 | 400 | 是 | 816 | 15,728,640 |
| F5 | 560 | 是 | 528 | 15,728,640 |

全模型都严格797,050,880参数；Q/K/V/O512独立、Identity中间激活，
原始QK norm/RoPE保留。UG中间Identity，原始SwiGLU SiLU保留；
Down使用无learnable gain的RMSNorm。各维度均为16的倍数。

FFN矩阵乘加/token与上述投影参数/层相同，五组都与P0匹配；这只统计
线性矩阵乘法。Down latent变宽会增加归一化工作和中间激活，共享会改变
分支执行方式，所以实际吞吐/显存仍须单独测量。

F1/F2与P0是预算分配对照；F3同时改变共享和分配，只能解释为组合候选。
P1(shared560/Down512)与F3之间也同时改变了两项rank，不能据此把收益
单独归因于Down。挑出收益后应补固定共享范围的分配曲线与重复种子。

与P0同初始化/数据种子12536/34521。全部30B、seq8192、globalbatch256，
microbatch32768、Muon.005、WSD2000warmup/末20%decay、FP8、1788步
checkpoint并保留全部。沿用LM验证和下游评估。主要指标最后1000步
平均Train CE，另保存完整均值/最终原始CE/验证CE，禁止混用短日程结果。

用户允许pool并行多个1x8低优任务；每30分钟监控。输出路径按run隔离。

## Q/K非线性与FFN预算重分配的组合

F1qk、F2qk、F3qk只将对应F1/F2/F3的Q、K中间激活由Identity改为SiLU：Q=SiLU(XA_q)B_q，K=SiLU(XA_k)B_k。V/O继续独立rank512 Identity，原始QK norm/RoPE不变。UG因子间继续Identity、保留原始SwiGLU，Down继续无gain RMSNorm。全模型均严格797,050,880参数，seq8192/global256/Muon0.005/WSD2000/30B/FP8与seed不变。

|配置|UG rank|UG共享|Down rank|
|---|---:|---|---:|
|F1qk|384|不共享|768|
|F2qk|256|不共享|1024|
|F3qk|480|共享|672|

分别与F1/F2/F3比较同step和完整Train CE；不能单凭一个末步CE选最优。已用局部配置变更验证检查只有Q/K/group变化，并核对真实meta模块参数量和8/16/32卡的globalbatch整除。launcher使用新的独立run路径、job-local编译缓存，并在GPU任务内对GPFS2checkpoint根做64GiB/16MiB写盘预检。

## 固定共享范围的预算分配曲线

F4qk、F5qk保持与F3qk相同的UG完全共享、Q/K SiLU、V/O Identity和
Down RMSNorm，只改变UG/Down的rank分配。这样可以在同一个共享背景
比较预算分配，而不是把共享变化和rank变化一起归因于Down。

|配置|UG共享rank|Down rank|全模型参数|
|---|---:|---:|---:|
|F5qk|560|528|797,050,880|
|F3qk|480|672|797,050,880|
|F4qk|400|816|797,050,880|

它们满足9*r_UG+5*r_Down=7680，FFN投影参数严格为15,728,640/层，
各维度为16的倍数。F5qk不同于少0.329%参数的P1：它将预算补到Down528。
两组已于2026-10-06 23:02/23:03提交arch2x8、PRIORITY=1。
23:08核实各2个active replicas、(dp_replicate=2,dp_shard=8)、step0已保存，
分别训练至step70/40，CE有限，无OOM/Traceback。精确W&B段与启动证据
已保存在jobs.json、startup_f4qk.log及startup_f5qk.log。

## 补齐共享FFN预算曲线的激活对照

新增F4（共享UG400/Down816）和F5（共享UG560/Down528），Q/K/V/O中间均Identity；其余严格沿用对应F4qk/F5qk：参数797,050,880、UG中间Identity/原生SwiGLU、Down无gain RMSNorm、seed12536/34521和完整30B日程。完成后分别检验这两个预算点的Q/K SiLU效果，并与F3/F3qk组成共享背景下的3×2曲线。此前模型与训练配置验证已覆盖这两组primary配置及其qk版本；新增入口和launcher已做语法检查。优先有空卡arch16/32，否则pool8，实际设备差异保留，不将多项收益直接相加。

F4/F5于2026-10-07 02:57分别提交有空卡的pool1x8低优。Arch16/32当时不可调度；对应QK组为arch16，设备数差异在jobs.json中保留。已有验证覆盖model/recipe；新增aligned-window对照测试1passed10.26s。
