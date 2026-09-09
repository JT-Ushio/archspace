"""Offline, exact-step analysis of the four Down activation snapshots."""
import csv
import json
import math
import statistics as st
import zipfile
from pathlib import Path

ROOT = Path(__file__).parent / "results_four_runs"
LABELS = ("S", "I", "R", "SR")
LOSS = "train/CE loss"


def finite(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def flat(d, pre=""):
    out = {}
    for k, v in d.items():
        key = f"{pre}.{k}" if pre else k
        out.update(flat(v, key) if isinstance(v, dict) else {key: v})
    return out


def csvout(name, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with (ROOT / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def table(rows, cols=None):
    if not rows:
        return "无共同记录。"
    cols = cols or list(rows[0])
    def fmt(v):
        return f"{v:.6f}" if isinstance(v, float) else str(v).replace("|", "/")
    return "\n".join(["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"] +
                     ["| " + " | ".join(fmt(row.get(k, "")) for k in cols) + " |" for row in rows])


def effects(v):
    return dict(remove_silu=v["S"]-v["I"], norm_without_silu=v["I"]-v["R"],
                norm_with_silu=v["S"]-v["SR"], silu_penalty_with_norm=v["SR"]-v["R"],
                total_s_to_r=v["S"]-v["R"],
                interaction=(v["S"]-v["SR"])-(v["I"]-v["R"]))


def main():
    meta = {l: json.loads((ROOT / f"{l}_metadata.json").read_text()) for l in LABELS}
    history = {l: [json.loads(line) for line in (ROOT / f"{l}_history.jsonl").read_text().splitlines()] for l in LABELS}
    # Merge any sparse history rows by step rather than discarding metrics.
    data = {}
    duplicates = {}
    for l in LABELS:
        data[l] = {}
        duplicates[l] = 0
        for row in history[l]:
            s = int(row["_step"])
            duplicates[l] += s in data[l]
            data[l].setdefault(s, {}).update({k: v for k, v in row.items() if v is not None})
    cfg = {l: flat(meta[l]["config"]) for l in LABELS}
    audit = [dict(field=k, **{l: json.dumps(cfg[l].get(k), sort_keys=True, ensure_ascii=False) for l in LABELS},
                  equal=len({json.dumps(cfg[l].get(k), sort_keys=True) for l in LABELS}) == 1)
             for k in sorted(set.union(*(set(v) for v in cfg.values())))]
    csvout("config_audit.csv", audit)
    differences = [r for r in audit if not r["equal"]]
    batch = cfg["S"]["data_loader.global_batch_size"]
    assert all(cfg[l]["data_loader.global_batch_size"] == batch for l in LABELS)
    token_errors = [(l, s, row["throughput/total tokens"]) for l in LABELS for s, row in data[l].items()
                    if finite(row.get("throughput/total tokens")) and abs(row["throughput/total tokens"]-s*batch)>1024]
    assert not token_errors, token_errors[:5]
    train_steps = {l: {s for s, row in data[l].items() if finite(row.get(LOSS))} for l in LABELS}
    common = sorted(set.intersection(*train_steps.values()))
    last = common[-1]
    bad = {s for s in common if any(data[l][s][LOSS] <= 0 for l in LABELS)}
    anomalies = [dict(run=l, step=s, loss=row.get(LOSS), z_loss=row.get("train/Z loss"),
                      ppl=row.get("train/PPL"), grad_norm=row.get("optim/total grad norm"))
                 for l in LABELS for s, row in data[l].items() if LOSS in row and
                 (not finite(row[LOSS]) or row[LOSS] <= 0)]
    csvout("anomalies.csv", anomalies or [dict(run="none", step="", loss="")])
    aligned = []
    for s in common:
        v = {l: data[l][s][LOSS] for l in LABELS}
        aligned.append(dict(step=s, tokens=s*batch, **v, **effects(v), anomalous=s in bad))
    csvout("aligned_training.csv", aligned)
    warmup = cfg["S"]["train_module.scheduler.warmup"]
    decay_start = last*(1-cfg["S"]["train_module.scheduler.decay_fraction"])
    windows = [("last_point", last-1, last), ("last_1pct", .99*last, last),
               ("last_5pct", .95*last, last), ("last_10pct", .9*last, last),
               ("last_20pct", .8*last, last), ("warmup", 0, warmup),
               ("stable", warmup, decay_start), ("decay", decay_start, last)]
    windows += [(f"stage_{i+1}_of_8", i*last/8, (i+1)*last/8) for i in range(8)]
    paired, distributions = [], []
    for policy in ("raw", "paired_positive_only"):
        for name, start, end in windows:
            steps = [s for s in common if start<s<=end and (policy=="raw" or s not in bad)]
            if not steps:
                continue
            v = {l: st.mean(data[l][s][LOSS] for s in steps) for l in LABELS}
            paired.append(dict(window=name, policy=policy, n=len(steps), first_step=steps[0], last_step=steps[-1], **v, **effects(v)))
            for l in LABELS:
                xs = [data[l][s][LOSS] for s in steps]
                distributions.append(dict(run=l, window=name, policy=policy, n=len(xs), mean=st.mean(xs),
                                          median=st.median(xs), std=st.pstdev(xs), min=min(xs), max=max(xs)))
    csvout("training_windows.csv", distributions)
    csvout("paired_effects.csv", paired)
    evalkeys = sorted({k for l in LABELS for row in data[l].values() for k, v in row.items()
                       if k.startswith("eval/lm/") and k.endswith("/CE loss") and finite(v)})
    evalsteps = sorted(set.intersection(*({s for s, row in data[l].items() if finite(row.get(k))}
                                         for l in LABELS for k in evalkeys)))
    validation, macro = [], []
    for s in evalsteps:
        for key in evalkeys:
            v = {l: data[l][s][key] for l in LABELS}
            validation.append(dict(dataset=key[len("eval/lm/"):-len("/CE loss")], step=s, tokens=s*batch, **v, **effects(v),
                                   **{f"{l}_ppl": math.exp(v[l]) for l in LABELS}))
        v = {l: st.mean(data[l][s][k] for k in evalkeys) for l in LABELS}
        macro.append(dict(step=s, tokens=s*batch, **v, **effects(v)))
    finaleval = evalsteps[-1]
    final = [r for r in validation if r["step"] == finaleval]
    csvout("validation_history.csv", validation)
    csvout("validation_final.csv", final)
    csvout("validation_macro.csv", macro)
    dskeys = sorted({k for l in LABELS for row in data[l].values() for k, v in row.items()
                     if k.startswith("eval/downstream/") and finite(v)})
    downstream = []
    for key in dskeys:
        steps = sorted(set.intersection(*({s for s, row in data[l].items() if finite(row.get(key))} for l in LABELS)))
        for s in steps:
            v = {l: data[l][s][key] for l in LABELS}
            downstream.append(dict(metric=key.removeprefix("eval/downstream/"), step=s, tokens=s*batch, **v))
    csvout("downstream_all_metrics.csv", downstream)
    selected_ds = [r for r in downstream if
                   ("_bpb_" in r["metric"] and r["metric"].endswith(" (BPB)")) or
                   ("_mc_" in r["metric"] and r["metric"].endswith((" (accuracy)", " (length-normalized accuracy)")))]
    csvout("downstream_selected.csv", selected_ds)
    operations, checkpoints, status = [], [], []
    opkeys = sorted({k for l in LABELS for row in data[l].values() for k, v in row.items() if finite(v) and
                     (k.startswith(("memory/", "throughput/device/")) or k in ("optim/total grad norm", "train/Z loss", "train/masked labels (%)"))})
    for l in LABELS:
        rows = data[l]
        status.append(dict(run=l, state=meta[l]["state"], last_step=max(train_steps[l]),
                           last_tokens=max(train_steps[l])*batch, train_rows=len(train_steps[l]),
                           missing_train_steps=len(set(range(1, max(train_steps[l])+1))-train_steps[l]),
                           duplicate_steps=duplicates[l], nonpositive_loss=sum(rows[s][LOSS]<=0 for s in train_steps[l]),
                           runtime_h=max((r.get("_runtime", 0) or 0) for r in rows.values())/3600))
        for key in opkeys:
            xs = [rows[s][key] for s in common if s>warmup and finite(rows[s].get(key))]
            if xs:
                operations.append(dict(run=l, metric=key, n=len(xs), mean=st.mean(xs), median=st.median(xs),
                                       p05=sorted(xs)[int(.05*(len(xs)-1))], p95=sorted(xs)[int(.95*(len(xs)-1))], max=max(xs)))
        for s, row in rows.items():
            for key, v in row.items():
                if key.startswith("checkpoint/") and finite(v):
                    checkpoints.append(dict(run=l, logged_step=s, metric=key, value=v))
    csvout("run_status.csv", status)
    csvout("operational_stats.csv", operations)
    csvout("checkpoint_events.csv", checkpoints)
    tail = next(r for r in paired if r["window"]=="last_10pct" and r["policy"]=="raw")
    conclusion = dict(last_train_step=last, last_eval_step=finaleval, tail=tail, validation=macro[-1],
                      validation_win_counts={e: sum(r[e]>0 for r in final) for e in effects(macro[-1])},
                      anomaly_count=len(anomalies), validation_points=len(macro), config_differences=differences)
    (ROOT / "summary.json").write_text(json.dumps(conclusion, ensure_ascii=False, indent=2))
    # Figures are descriptive trajectories, not independent-seed uncertainty intervals.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from collections import deque
    def rolling(values, width=100):
        q, out = deque(maxlen=width), []
        for v in values:
            q.append(v)
            valid = [x for x in q if finite(x)]
            out.append(st.mean(valid) if valid else float("nan"))
        return out
    colors = dict(S="#ce6d28", I="#2875b9", R="#25905d", SR="#9a4bb6")
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for l in LABELS:
        axes[0, 0].plot([s*batch/1e9 for s in common], rolling([data[l][s][LOSS] if s not in bad else float("nan") for s in common]), color=colors[l], label=l)
        axes[1, 0].plot([r["tokens"]/1e9 for r in macro], [r[l] for r in macro], color=colors[l], label=l)
    for effect in ("remove_silu", "norm_without_silu", "norm_with_silu", "silu_penalty_with_norm"):
        axes[0, 1].plot([r["tokens"]/1e9 for r in aligned], rolling([r[effect] if not r["anomalous"] else float("nan") for r in aligned]), label=effect)
        axes[1, 1].plot([r["tokens"]/1e9 for r in macro], [r[effect] for r in macro], label=effect)
    for a, title in zip(axes.flat, ("Train CE, trailing 100 logged steps", "Train paired CE differences", "Validation CE, 11-dataset equal mean", "Validation paired CE differences")):
        a.set_title(title); a.set_xlabel("Training tokens (B)"); a.grid(alpha=.25); a.legend(fontsize=8)
    axes[0, 0].set_ylim(2.45, 4.0)
    for a in (axes[0, 1], axes[1, 1]):
        a.axhline(0, color="black", linewidth=.8)
    fig.tight_layout(); fig.savefig(ROOT/"overview.png", dpi=180); fig.savefig(ROOT/"overview.pdf"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 7))
    for j, effect in enumerate(("remove_silu", "norm_without_silu", "norm_with_silu")):
        ax.barh([i+(j-1)*.25 for i in range(len(final))], [r[effect] for r in final], height=.24, label=effect)
    ax.set_yticks(range(len(final)), [r["dataset"] for r in final]); ax.axvline(0, color="black", linewidth=.8)
    ax.set_xlabel("CE difference (positive = benefit)"); ax.set_title(f"Validation effects, step {finaleval}"); ax.legend(); fig.tight_layout()
    fig.savefig(ROOT/"validation_effects.png", dpi=180); plt.close(fig)
    topline = [dict(run=l, tail10_train_ce=tail[l], last_train_ce=data[l][last][LOSS], validation_macro_ce=macro[-1][l]) for l in LABELS]
    csvout("headline.csv", topline)
    ratio = 100*macro[-1]["remove_silu"]/macro[-1]["total_s_to_r"]
    bpb_rows = [r for r in selected_ds if r['metric'].endswith(' (BPB)')]
    mmlu_acc = [r for r in selected_ds if r['metric'].startswith('mmlu_') and r['metric'].endswith(' (length-normalized accuracy)')]
    cols = ["window", *LABELS, *effects(tail)]
    report = f'''# Down 激活四组 FP8 / Muon / 30B 实验报告

## 1. 主要结果

S=SiLU，I=Identity，R=RMSNorm，SR=RMSNorm(SiLU(a))。主指标中观察到 **I≈R < S≈SR**。
最新共同验证 step {finaleval}（{finaleval*batch/1e9:.6f}B），不是最终训练 step {last} 的重评。

{table(topline)}

移除 SiLU 的验证 CE 收益 S−I={macro[-1]['remove_silu']:.6f}；I 基础上加入 RMSNorm 的额外收益 I−R={macro[-1]['norm_without_silu']:.6f}。
S→R 总差值为 {macro[-1]['total_s_to_r']:.6f}，其中 S→I 占 {ratio:.2f}%。这是本次观测差值的算术分解，不是因果贡献率或跨种子结论。
保留 SiLU 再做 RMSNorm：S−SR={macro[-1]['norm_with_silu']:.6f}，负值表示 SR 反而略差。
因此本次结果支持「主要收益来自移除 Down 的额外 SiLU」，不支持「只要把 SiLU 后幅度归一化就能获得相同提升」。

![曲线总览](overview.png)

## 2. 运行完成和配置核对

{table(status)}

每个有 token 计数的历史记录都核对了 tokens≈step×{batch:,}（容许 1024 tokens 的日志舍入）；比较使用真实共同 step，没有把不同长度的曲线重新缩放。
完整配置逐字段核对见 config_audit.csv。实际不同字段：

{table(differences, ['field', *LABELS])}

共同设置：global batch 256×8192、30B token budget、Muon LR=0.005、WSD warmup=2000 / decay_fraction=0.2、FP8 enabled、microbatch=32768、所有 rank=512、Gate/Up 不共享、Q/K/V/O 不共享且 SiLU。
Down 内部 RMSNorm 不带 gain，r1=down_r1=0；原有 FFN 输出 RMSNorm 保留。数据 seed / 初始化 seed / 其他训练配置见完整配置审计。
配置匹配不能证明每次运行的源码 commit、GPU 型号、kernel 或初始化张量逐元素相同，本次未读取权重文件。

## 3. 训练阶段与末段窗口

{table([r for r in paired if r['policy']=='raw' and r['window'].startswith('last')], cols)}

{table([r for r in paired if r['policy']=='paired_positive_only' and (r['window'].startswith('stage') or r['window'] in ('warmup','stable','decay'))], cols)}

定义 remove_silu=S−I；norm_without_silu=I−R；norm_with_silu=S−SR；silu_penalty_with_norm=SR−R；interaction=(S−SR)−(I−R)。
所有差值用同一组 step 配对计算。均值是记录点算术均值；本次缺失数见运行表。标准差只描述训练阶段/批次波动，不是跨 seed 误差条。
发现异常 CE（非有限或≤0）记录 {len(anomalies)} 条，原始记录保留在 JSONL 和 anomalies.csv。raw 表不删除有限零值；paired_positive_only 版本在任一组异常时四组一起排除。非有限 CE 不参与均值。
具体异常：{table(anomalies)}
R 的 step 10728 正好是周期保存位置，CE=0、PPL=1、Z-loss=0 但 grad norm 非零；仅能标记为可疑记录，不能证明发生了真实的零损失优化步，也不能确认一定由保存逻辑导致。它不位于最后 10% 窗口。
该单点会影响小差距的方向：stable 段原始 I−R 约 +0.000162，四组共同排除该 step 后约 −0.000114。因此不要用含该点的整段均值声称 RMSNorm 比 Identity 有确定优势。
图为 trailing-100 点平滑，异常点共同排除；训练绝对 CE 图显示 2.45–4.0 的局部范围，完整起始值见原始数据。

## 4. 验证集与阶段性

共 {len(evalkeys)} 个常规 LM 验证集，{len(evalsteps)} 个共同评测点，范围 step {evalsteps[0]}–{finaleval}。
以下均值为跨数据集等权 CE，不是按 tokens 加权的总体 CE，也不是 BPB。

{table(final, ['dataset', *LABELS, 'remove_silu','norm_without_silu','norm_with_silu','silu_penalty_with_norm'])}

{table(macro, ['step', *LABELS, 'remove_silu','norm_without_silu','norm_with_silu'])}

最后一次验证中，移除 SiLU 在 {sum(r['remove_silu']>0 for r in final)}/{len(final)} 集上改善；I→R 在 {sum(r['norm_without_silu']>0 for r in final)}/{len(final)} 集上改善；S→SR 在 {sum(r['norm_with_silu']>0 for r in final)}/{len(final)} 集上改善。
跨验证时间点：S−I 在 {sum(r['remove_silu']>0 for r in macro)}/{len(macro)} 个共同点为正；I−R 在 {sum(r['norm_without_silu']>0 for r in macro)}/{len(macro)} 点为正；S−SR 仅在 {sum(r['norm_with_silu']>0 for r in macro)}/{len(macro)} 点为正。
所以主分组排序跨进度稳定，但 I/R 的微小差距不支持“内部 RMSNorm 在所有域都更好”。
每个验证集的 PPL 由 exp(该集 CE) 得到，见 validation_final.csv。所有共同时间点和效应见 validation_history.csv / validation_macro.csv。

![逐验证集效果](validation_effects.png)

## 5. 下游评测（独立于常规 LM 验证）

以下是 W&B 实际记录的 BPB / accuracy，不是由 CE 换算的 BPB。BPB 越低越好，accuracy 越高越好。
这些下游结果在表中各自的 step 产生，不能称为 30B 最终 checkpoint 的重新评估。
只选任务名称带 _bpb_ 的原始 BPB，以及 _mc_ 的原始 accuracy / length-normalized accuracy（表中明确区分），避免把 v2 指标重复计作独立实验；全部指标、包括 v2 和其他任务，都保留在 downstream_all_metrics.csv。

{table(selected_ds, ['metric','step',*LABELS])}

任务格式和尺度不同，不把 BPB 和 accuracy 混合成一个总分；也不把相关的同任务 v1/v2 视为独立样本。
这里下游结果没有统一复现主 CE 排序：R 的 BPB 在 {sum(r['R']<r['I'] for r in bpb_rows)}/{len(bpb_rows)} 个选定 BPB 任务优于 I，但 MMLU 的 {sum(r['R']<r['I'] for r in mmlu_acc)}/{len(mmlu_acc)} 个 length-normalized accuracy 指标反而低于 I。
例如 MMLU STEM BPB：I=1.795935、R=1.767170；对应 MC length-normalized accuracy：I=26.94%、R=26.21%。Gold-answer BPB 与选择题准确率衡量的目标不同，这些准确率本身偏低，不能把 LM loss 改善直接说成所有下游能力提升，也不能在未核验评测实现前断言模型退化或评测有 bug。

## 6. 梯度、吞吐和 checkpoint

{table([r for r in operations if r['metric'] in ('optim/total grad norm','throughput/device/TPS')], ['run','metric','n','mean','median','p05','p95','max'])}

统计采用共同 step>{warmup} 的有效记录。吞吐受卡数、拓扑、共享负载和评测影响，不能直接当作受控架构加速比；全部可用 memory/throughput 指标见 operational_stats.csv。
checkpoint 日志事件 {len(checkpoints)} 条，见 checkpoint_events.csv。save_interval 配置为1788，异步完成日志的 step 不一定就是 checkpoint 的保存 step；初始和结束也可能额外保存。日志有事件不能证明磁盘 ckpt 完整，本次未做 ckpt 文件完整性检查。

## 7. 机制解释与后续

四组构成「有无 SiLU × 有无内部 RMSNorm」。对单分支无 bias 的读出，忽略 epsilon 的精确算术下，外部 RMSNorm 会抵消内部正尺度，所以 I/R 和 S/SR 分别应理想等价。
实际结果大致按这两对聚类，支持移除额外 SiLU 比内部尺度归一化更重要。SR 没有恢复到 I/R 水平，说明在这个预算下仅归一化 SiLU 后的输出不足以消除差距。
这不是证明 FP8 数值效应不存在：小的 I/R、S/SR 差距仍可能来自量化、epsilon、舍入、优化轨迹或随机波动，也可能与特定实现有关；仅凭本实验无法归因于长 K Tensor Core 累加。
建议先复用已有 AdamW-S/R 结果检查激活收益是否依赖优化配方；再为 I/R 或 S/I 做配对 seeds，以检验小差异的稳定性。若要证明数值机制，再用同 checkpoint / 同输入做逐层离线前向及梯度误差测量。不要仅凭最后一条 train loss 决策。

## 8. 可复现数据

原始 metadata/JSONL、配置审计、逐步训练、窗口统计、逐验证集/逐评测时点、下游全指标、运行开销、checkpoint 事件、异常数据以及图已保存在本目录。
本次操作仅为 W&B 只读导出和本地分析，没有修改远端 run、训练配置或 checkpoint。
'''
    report += "\nW&B 来源：\n\n" + "\n".join(f"- [{l}: {meta[l]['name']}]({meta[l]['url']})" for l in LABELS) + "\n"
    (ROOT / "report.md").write_text(report)
    with zipfile.ZipFile(ROOT / "down_activation_report.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(ROOT.iterdir()):
            if path.is_file() and path.suffix != ".zip":
                archive.write(path, path.name)
    print(json.dumps(conclusion, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
