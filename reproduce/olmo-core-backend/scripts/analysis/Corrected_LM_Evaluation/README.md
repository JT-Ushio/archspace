# 独立复评：按有效 target 统计 LM CE

当前 OLMo-core 的 `get_labels` 先应用输入 `label_mask`，再将 labels 左移并在末位填 ignore_index；`LMEvaluator.update_metrics` 却直接用未移位的输入 mask 选择逐 token CE。这可能纳入 ignore target 的零 loss，并在 prefix mask 下漏掉本应评分的 target。该问题不改写本项目的 Train CE 排名。

本目录仅在独立 GPU 评估进程中适配 `eval_batch`：forward 使用原输入/labels；forward 完成后，用 `labels != label_ignore_index` 评分，同时保留旧 mask 的分数。两种指标来自同一 checkpoint、同一数据、同一 forward，并记录每个源的整数 token 数。分布式 reduce 使用副本，重复收集不会改变累加器。只支持当前 DP 场景，明确拒绝 CP/TP。

首批复评已完成的 P0qk/F3qk，各自读取精确 `step14306`，统一 pool8、seq8192、原始 FP8/模型配置和11源完整 LM 验证。构造后直接调用 evaluator，禁止训练、optimizer step 与 checkpoint save；不调用 `trainer.fit()`，不恢复 optimizer/trainer/RNG状态，不更改训练脚本或 upstream OLMo-core。临时工作文件使用 job-local tmp，结果只输出 JSON/SHA256 与临时文件。

这次是 final14306 checkpoint；旧 W&B 验证来自 step14000，不能把两者差值全部归因于 mask。只有同次 forward 的 legacy/corrected 两列才能隔离评分口径差异。Train CE 仍为主要选型标准。

CPU验证：padding、prefix、无mask末位、序列并行拒绝、重复分布式计数，共5tests passed59.99s。首批真实8卡执行已确认完成，两个checkpoint加载、11源结果及各自JSON/SHA256均已核实，见[结果](RESULTS.md)。历史C0/C2/C6的实际路径另从原始W&B配置解析，并与最终config.json核对，避免按当前launcher猜测旧目录；后续复评还核对完整model配置及init_seed。

补充：原始Dense训练使用65536-token microbatch；独立复评明确统一为32768，与低秩模型的数据batch划分相同。Dense实际checkpoint路径由两个精确W&B续训片段及最终metadata解析，不猜目录，原始调度job ID未知时明确记为null。评估前校验resolver任务已完成、payload SHA256、完整14306历史及源run身份，再核对checkpoint model/seed并实际加载。新增来源证据测试5passed0.16s。
