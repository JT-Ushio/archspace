"""Build a reproducible report from export_four_runs.py's local snapshots."""
import csv
import json
import math
import statistics as st
import zipfile
from pathlib import Path

LABELS = ('bf16_silu', 'bf16_rms', 'fp8_silu', 'fp8_rms')
ROOT = Path(__file__).parent / 'results_four_runs'
LOSS = 'train/CE loss'


def finite(x):
    return isinstance(x, (float, int)) and not isinstance(x, bool) and math.isfinite(x)


def flat(d, prefix=''):
    out = {}
    for k, v in d.items():
        key = prefix + '.' + k if prefix else k
        out.update(flat(v, key) if isinstance(v, dict) else {key: v})
    return out


def csvout(name, rows):
    if not rows:
        return
    with (ROOT / name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def table(rows, cols=None):
    cols = cols or list(rows[0])
    def fmt(x):
        return f'{x:.6f}' if isinstance(x, float) else str(x).replace('|', '/')
    return '\n'.join(['| ' + ' | '.join(cols) + ' |', '| ' + ' | '.join(['---']*len(cols)) + ' |'] +
                     ['| ' + ' | '.join(fmt(r[c]) for c in cols) + ' |' for r in rows])


def effects(values):
    db = values['bf16_silu'] - values['bf16_rms']
    df = values['fp8_silu'] - values['fp8_rms']
    return dict(delta_bf16=db, delta_fp8=df, interaction=df-db,
                fp8_penalty_silu=values['fp8_silu']-values['bf16_silu'],
                fp8_penalty_rms=values['fp8_rms']-values['bf16_rms'])


def quantile(xs, q):
    xs = sorted(xs)
    i = (len(xs)-1)*q
    a, b = math.floor(i), math.ceil(i)
    return xs[a] + (xs[b]-xs[a])*(i-a)


def main():
    meta = {l: json.loads((ROOT / f'{l}_metadata.json').read_text()) for l in LABELS}
    histories = {l: [json.loads(x) for x in (ROOT / f'{l}_history.jsonl').read_text().splitlines()] for l in LABELS}
    data = {l: {int(r['_step']): r for r in histories[l] if finite(r.get(LOSS))} for l in LABELS}
    cfg = {l: flat(meta[l]['config']) for l in LABELS}
    common = sorted(set.intersection(*(set(data[l]) for l in LABELS)))
    last = common[-1]
    batch = cfg[LABELS[0]]['data_loader.global_batch_size']
    assert all(cfg[l]['data_loader.global_batch_size'] == batch for l in LABELS)
    # Verify that W&B step really maps to optimizer tokens, rather than normalize runs independently.
    token_errors = []
    for l in LABELS:
        for s, r in data[l].items():
            tok = r.get('throughput/total tokens')
            if finite(tok) and abs(tok-s*batch) > max(1024, tok*1e-6):
                token_errors.append((l,s,tok))
    assert not token_errors, token_errors[:5]
    assert all(max(data[l]) == last for l in LABELS), 'Unequal run lengths'
    bad = sorted(s for s in common if any(data[l][s][LOSS] <= 0 for l in LABELS))
    anomalies = [dict(run=l,step=s,loss=data[l][s][LOSS],ppl=data[l][s].get('train/PPL'),
                      z_loss=data[l][s].get('train/Z loss'),grad_norm=data[l][s].get('optim/total grad norm'))
                 for l in LABELS for s in common if data[l][s][LOSS] <= 0]
    csvout('anomalies.csv', anomalies)
    aligned=[]
    for s in common:
        v={l:data[l][s][LOSS] for l in LABELS}
        aligned.append(dict(step=s,tokens=s*batch,**v,**effects(v),nonpositive_any=s in bad))
    csvout('aligned_training.csv', aligned)
    windows=[('last_point',last-1,last), ('last_1pct',.99*last,last), ('last_5pct',.95*last,last),
             ('last_10pct',.9*last,last), ('last_20pct',.8*last,last),
             ('warmup',0,500),('stable',500,.8*last),('decay',.8*last,last),('full',0,last)]
    windows += [(f'quarter_{i+1}',i*last/4,(i+1)*last/4) for i in range(4)]
    windows += [(f'checkpoint_{s}_trailing50',s-50,s) for s in (894,1788,2682,3576)]
    stats=[]; paired=[]
    for policy in ('raw','paired_positive_only'):
        for name,start,end in windows:
            steps=[s for s in common if start<s<=end and (policy=='raw' or s not in bad)]
            v={l:st.mean(data[l][s][LOSS] for s in steps) for l in LABELS}
            paired.append(dict(window=name,policy=policy,n=len(steps),first_step=steps[0],last_step=steps[-1],**v,**effects(v)))
            for l in LABELS:
                xs=[data[l][s][LOSS] for s in steps]
                stats.append(dict(run=l,window=name,policy=policy,n=len(xs),mean=st.mean(xs),
                                  median=st.median(xs),std=st.pstdev(xs),p05=quantile(xs,.05),p95=quantile(xs,.95),
                                  minimum=min(xs),maximum=max(xs),exp_mean_ce=math.exp(st.mean(xs))))
    csvout('training_windows.csv',stats); csvout('paired_effects.csv',paired)
    # Full configuration audit, including fields which differ only for run bookkeeping.
    audit=[dict(field=k,**{l:json.dumps(cfg[l].get(k),ensure_ascii=False) for l in LABELS},
                equal=len({json.dumps(cfg[l].get(k),sort_keys=True) for l in LABELS})==1)
           for k in sorted(set.union(*(set(c) for c in cfg.values())))]
    csvout('config_audit.csv',audit)
    diff=[r for r in audit if not r['equal']]
    evalkeys=sorted({k for rows in histories.values() for r in rows for k,v in r.items()
                     if k.startswith('eval/lm/') and k.endswith('/CE loss') and finite(v)})
    evalmaps={l:{k:{int(r['_step']):r[k] for r in histories[l] if finite(r.get(k))} for k in evalkeys} for l in LABELS}
    evalsteps=sorted(set.intersection(*(set(evalmaps[l][k]) for l in LABELS for k in evalkeys)))
    finaleval=evalsteps[-1]
    evaluations=[]; evalhistory=[]
    for s in evalsteps:
        for k in evalkeys:
            dataset=k[len('eval/lm/'):-len('/CE loss')]
            v={l:evalmaps[l][k][s] for l in LABELS}
            row=dict(dataset=dataset,step=s,tokens=s*batch,**v,**effects(v))
            evalhistory.append(row)
            if s==finaleval:
                evaluations.append(dict(**row,**{l+'_ppl':math.exp(v[l]) for l in LABELS},
                                        bf16_rms_ppl_reduction_pct=100*(1-math.exp(-row['delta_bf16'])),
                                        fp8_rms_ppl_reduction_pct=100*(1-math.exp(-row['delta_fp8']))))
    csvout('validation_history.csv',evalhistory); csvout('validation_final.csv',evaluations)
    macro=[]
    for s in evalsteps:
        v={l:st.mean(evalmaps[l][k][s] for k in evalkeys) for l in LABELS}
        macro.append(dict(step=s,tokens=s*batch,**v,**effects(v)))
    csvout('validation_macro.csv',macro)
    operational=[]; ckpt=[]; quality=[]
    metrics=['optim/total grad norm','throughput/device/TPS','throughput/device/TPS (actual avg)',
             'throughput/device/MFU','train/Z loss','train/masked labels (%)']
    for l in LABELS:
        rows=histories[l]
        quality.append(dict(run=l,state=meta[l]['state'],history_rows=len(rows),loss_rows=len(data[l]),
                            last_step=max(data[l]),last_tokens=data[l][last].get('throughput/total tokens'),
                            missing_loss_steps=len(set(range(1,last+1))-set(data[l])),
                            nonpositive_loss_count=sum(r[LOSS]<=0 for r in data[l].values()),
                            runtime_s=max(r['_runtime'] for r in rows if finite(r.get('_runtime')))))
        for key in metrics:
            xs=[r[key] for r in rows if r['_step']>500 and finite(r.get(key))]
            if xs: operational.append(dict(run=l,metric=key,window='step>500',n=len(xs),mean=st.mean(xs),median=st.median(xs),
                                           p05=quantile(xs,.05),p95=quantile(xs,.95),maximum=max(xs)))
        for r in rows:
            for k in ('checkpoint/save_async_duration_s','checkpoint/save_duration_s'):
                if finite(r.get(k)):
                    ckpt.append(dict(run=l,logged_step=r['_step'],metric=k,duration_s=r[k]))
        history_keys = sorted(set().union(*(set(x) for x in rows)))
        csvout(l+'_history.csv',[{k:r.get(k) for k in history_keys} for r in rows])
    csvout('run_status.csv',quality); csvout('operational_stats.csv',operational); csvout('checkpoint_events.csv',ckpt)
    # Plots use exact common optimizer steps. Invalid zero values form gaps only in figures.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    colors=['#2366b3','#0a8b67','#db7823','#ab3991']
    def rolling(values, width=50):
        return [st.mean(v for v in values[max(0,i-width+1):i+1] if finite(v))
                if any(finite(v) for v in values[max(0,i-width+1):i+1]) else float('nan') for i in range(len(values))]
    x=np.array(common)*batch/1e9
    fig,ax=plt.subplots(2,2,figsize=(13,9))
    for l,c in zip(LABELS,colors):
        ys=rolling([data[l][s][LOSS] if s not in bad else float('nan') for s in common])
        ax[0,0].plot(x,ys,label=l,color=c); ax[0,0].set_ylim(2.5,4.5)
        ax[1,0].plot([r['tokens']/1e9 for r in macro],[r[l] for r in macro],label=l,color=c,marker='o')
    for key,c in [('delta_bf16','#2366b3'),('delta_fp8','#db7823'),('interaction','#ab3991')]:
        ax[0,1].plot(x,rolling([r[key] if r['step'] not in bad else float('nan') for r in aligned]),label=key,color=c)
        ax[1,1].plot([r['tokens']/1e9 for r in macro],[r[key] for r in macro],label=key,color=c,marker='o')
    for a,title in zip(ax.flat,['Training CE: trailing 50-step mean (zoomed)','Training RMS benefit and interaction',
                               f'Validation CE: unweighted mean over {len(evalkeys)} datasets','Validation RMS benefit and interaction']):
        a.set_title(title); a.set_xlabel('Training tokens (B)'); a.set_ylabel('CE / CE difference'); a.grid(alpha=.25); a.legend(fontsize=8)
    for a in (ax[0,1],ax[1,1]): a.axhline(0,color='black',linewidth=.8)
    fig.tight_layout(); fig.savefig(ROOT/'overview.png',dpi=180); fig.savefig(ROOT/'overview.pdf'); plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,7))
    y=np.arange(len(evaluations))
    ax.barh(y-.18,[r['delta_bf16'] for r in evaluations],height=.35,label='BF16')
    ax.barh(y+.18,[r['delta_fp8'] for r in evaluations],height=.35,label='FP8')
    ax.set_yticks(y,[r['dataset'] for r in evaluations]); ax.axvline(0,color='black',linewidth=.8)
    ax.set_xlabel('SiLU CE - RMSNorm CE (positive favors RMSNorm)'); ax.set_title(f'Validation at step {finaleval}')
    ax.legend(); ax.grid(axis='x',alpha=.2); fig.tight_layout(); fig.savefig(ROOT/'validation_benefit.png',dpi=180); plt.close(fig)
    tail=next(r for r in paired if r['window']=='last_10pct' and r['policy']=='raw')
    selected=[r for r in paired if r['policy']=='raw' and r['window'] in ('last_point','last_1pct','last_5pct','last_10pct','last_20pct')]
    stage=[r for r in paired if r['policy']=='paired_positive_only' and (r['window'].startswith('quarter_') or r['window'] in ('warmup','stable','decay'))]
    cols=['window',*LABELS,'delta_bf16','delta_fp8','interaction']
    lastmacro=macro[-1]
    configkeys=['model.d_model','model.n_layers','model.init_seed','init_seed','data_loader.seed',
                'data_loader.global_batch_size','train_module.rank_microbatch_size','train_module.max_sequence_length',
                'train_module.optim.lr','train_module.optim.weight_decay','train_module.optim.betas','train_module.max_grad_norm',
                'train_module.scheduler.warmup','train_module.scheduler.decay_fraction','train_module.dp_config.param_dtype',
                'train_module.dp_config.reduce_dtype','train_module.compile_model','trainer.max_duration.value',
                'trainer.callbacks.checkpointer.save_interval','trainer.callbacks.checkpointer.max_checkpoints',
                'model.block.sequence_mixer.rank','model.block.feed_forward.r2','model.block.feed_forward.down_r2',
                'model.block.sequence_mixer.n_heads','model.block.sequence_mixer.backend','model.block.sequence_mixer.share_scope',
                'model.block.sequence_mixer.q_nonlinearity','model.block.sequence_mixer.k_nonlinearity',
                'model.block.sequence_mixer.v_nonlinearity','model.block.sequence_mixer.o_nonlinearity',
                'model.block.feed_forward.hidden_size','model.block.feed_forward.r1','model.block.feed_forward.down_r1',
                'model.block.feed_forward.gate_nonlinearity','model.block.feed_forward.up_nonlinearity',
                'model.block.feed_forward.rms_beta_scale','train_module.optim.mu','train_module.optim.adjust_lr',
                'model.block.feed_forward.share_gate_up_subspace','model.block.feed_forward.rms_norm_learnable_weight']
    configtable=[dict(field=k,value=cfg[LABELS[0]][k]) for k in configkeys if k in cfg[LABELS[0]]]
    performance=[r for r in operational if r['metric'] in ('throughput/device/TPS','optim/total grad norm')]
    report=f'''# Down 激活 × 训练精度：四组 7.5B 实验完整报告

## 1. 核心结果

四组均已完成。最后 10% 的训练 CE 中，BF16 的 RMSNorm 收益为 **{tail['delta_bf16']:.6f}**，
FP8 为 **{tail['delta_fp8']:.6f}**，交互项 **I={tail['interaction']:+.6f}**。
最后一次共同验证（step {finaleval}，{finaleval*batch/1e9:.6f}B tokens）在 {len(evalkeys)} 个数据集上的等权平均 CE：
BF16 收益 **{lastmacro['delta_bf16']:.6f}**、FP8 收益 **{lastmacro['delta_fp8']:.6f}**、交互项 **{lastmacro['interaction']:+.6f}**。

在这一次种子上，Down-RMSNorm 在两种精度下都更好，FP8 额外放大的幅度相对有限。
训练末段 FP8 收益约比 BF16 大 {100*tail['interaction']/tail['delta_bf16']:.2f}%，
验证等权平均约大 {100*lastmacro['interaction']/lastmacro['delta_bf16']:.2f}%。
这些百分比是“收益差距的相对变化”，不是 loss 下降了同样的百分比。
由此不支持“RMSNorm 的优势只来自 FP8”的强解释；更符合精度之外的激活/归一化收益为主、
同时存在较小精度交互的观察。当前尚不能确定额外交互是否跨种子稳定，更不能将其直接归因于长 K 累加。

四组末段及验证平均的排序均为 BF16-RMSNorm、FP8-RMSNorm、BF16-SiLU、FP8-SiLU（从低 loss 到高 loss）。
FP8-RMSNorm 甚至优于 BF16-SiLU，说明本预算下激活选择的影响大于记录到的 FP8 精度代价。
这不表示 FP8 没有代价：同激活比较仍显示 FP8 平均 CE 更高。

这些数值的含义需结合训练阶段、验证集与单种子限制，不能单凭 FP8 下 RMSNorm 胜出就证明长 K 累加机制。

![训练与验证总览](overview.png)

## 2. 实验问题与统计定义

四组是 BF16/FP8 × Down-SiLU/Down-RMSNorm。只改变 Down 中间激活以及 Float8 转换开关。
定义 Δ_BF16=L(BF16,SiLU)−L(BF16,RMS)，Δ_FP8=L(FP8,SiLU)−L(FP8,RMS)，I=Δ_FP8−Δ_BF16。
Δ>0 表示 RMSNorm 更好；I>0 表示该指标下 FP8 放大收益。另导出同激活下 FP8−BF16 的 CE 差值，正值表示 FP8 代价。

这里 FP8 指框架启用 Float8 的线性层转换策略，不表示 Norm、softmax、参数存储和所有算子均为 FP8。
此 2×2 比较涉及完整前向和反向训练轨迹，不能单独分离输入量化、Tensor Core 累加、反向量化及优化轨迹影响。

## 3. 实际配置与数据核验

{table(quality)}

W&B `_step` 与 `throughput/total tokens / global_batch_size` 已逐条交叉检查（允许日志浮点舍入），
四组在相同 optimizer step 上比较，没有按各自曲线长度重新缩放。目标 7.5B，在完整 batch 边界结束得到 {last*batch:,} tokens。
各组有 3578 行历史，含 step 0；有训练 loss 的记录为 3577 条。相同随机种子不保证不同架构的权重逐元素对应。

共同超参数：

{table(configtable)}

模型为 16 层、d_model=2048、FFN hidden_size=8192、16 heads（每 head 128 维）。
Q/K/V/O 各自 rank=512 且 share_scope=none，Gate/Up 不共享且 rank=512；Down 为 8192→512→2048。
Q/K/V/O 和 Gate/Up 的中间激活均为 SiLU；Down 的 SiLU/RMSNorm 为实验变量，RMS 可学习权重关闭。
因此 Down encoder 的前向 GEMM reduction K=8192 是本结构的事实，日志中的 rank 是 512，而非早期讨论举例的 256。

全部配置叶子比较见 [config_audit.csv](config_audit.csv)。不同的字段如下（名称、保存路径是运行标识差异）：

{table(diff, ['field',*LABELS])}

本次未核验 GPU 型号/拓扑、运行时代码 commit 和实际 kernel 精度；配置一致不能替代这些运行时证据。

## 4. 训练 CE：末段及各阶段

原始记录的末段算术平均（不删除零值；这些末段窗口不包含 step 894）：

{table(selected,cols)}

本报告采用共同 step 区间内逐 step 算术平均，不插值、不做时间加权。旧脚本是梯形积分均值，
因此末位小数可能不同。窗口为 start < step ≤ end；完整窗口边界和样本数见 CSV。
exp(mean CE) 仅作为该平均 CE 对应的 perplexity，不能解释为训练过程中 PPL 的算术平均。
标准差、5%/95% 分位数描述随训练/批次变化，不能作为独立重复实验误差条。

阶段统计（敏感性版本：遇到任一组 CE≤0，就从四组同时排除该 step，保持配对）：

{table(stage,cols)}

完整原始版与配对排除版均保存在 [paired_effects.csv](paired_effects.csv)；
[training_windows.csv](training_windows.csv) 包含均值、中位数、标准差、分位数、最小/最大值，
并提供四个 checkpoint 位置前 50 steps 的窗口统计。图中为 trailing-50 平滑，异常 step 在四组窗口中共同排除，
窗口仍用其余有效点求均值，故图中不会展示原始零值尖峰。训练图纵轴放大至 2.5–4.5，启动阶段较高 loss 不在视窗内；原始数据完整保留。

轨迹具有阶段性：warmup 平均 BF16 的 RMS 收益为负，进入稳定期后转为约 0.022–0.024。
第 2 个四分之一阶段的 I 略为负，后半程 I 转为小幅正值。因此最终的正交互不是贯穿全程、恒定幅度的现象。

## 5. 验证集结果

验证共同记录 step 为 {evalsteps}。最后一次验证是 step {finaleval}，不是最终 step {last}；
不能把此表称为最终 checkpoint 的重新评估。{len(evalkeys)} 个数据集等权平均仅作描述，未按各数据集 token 数加权。

{table(evaluations,['dataset',*LABELS,'delta_bf16','delta_fp8','interaction'])}

最终共同验证的等权汇总及各阶段：

{table(macro,['step',*LABELS,'delta_bf16','delta_fp8','interaction'])}

RMSNorm 在 BF16 的 {sum(r['delta_bf16']>0 for r in evaluations)}/{len(evaluations)} 个验证集上 CE 更低；
在 FP8 的 {sum(r['delta_fp8']>0 for r in evaluations)}/{len(evaluations)} 个验证集上更低。
交互项在 {sum(r['interaction']>0 for r in evaluations)}/{len(evaluations)} 个数据集上为正；
其余数据集为负，表明 FP8 放大效应并非每个语料域都一致。
跨验证时间点也有变化：step 1500、2000、3000 的等权 I 为负，最终 3500 为正。
最终截面的正交互应视为本次末段观测，而非已经证实的普遍 FP8 机制。
各集 PPL、PPL 相对降低百分比和同激活 FP8 代价见 [validation_final.csv](validation_final.csv)，
所有验证时间点见 [validation_history.csv](validation_history.csv)。

![各验证集 RMSNorm 收益](validation_benefit.png)

## 6. 梯度、吞吐与 checkpoint

排除前 500 step 后的统计：

{table(performance,['run','metric','n','mean','median','p05','p95','maximum'])}

TPS 是日志中的每设备指标；没有确认卡数/拓扑及机器负载，因此这些观测值不能当成受控 FP8 加速比。
总 runtime 包含启动、编译、评估、保存和可能的等待，不等于纯训练时间。
梯度范数来自 `optim/total grad norm`；是否为裁剪前值未通过运行时代码核验。
两种精度下 RMSNorm 的记录梯度范数均值均较 SiLU 低，但较低范数本身不能证明累加误差得到修复。
四组吞吐中位数都约在 88k tokens/s/device，未呈现大幅 FP8 吞吐优势；是否受 kernel、低秩小矩阵、
通信或评估开销限制，需另行性能剖析。
没有导出可用的显存峰值记录，因此不报告推测的显存节省。

checkpoint 配置间隔为 894、保留上限 4。保存耗时事件明细如下：

{table(ckpt)}

异步保存完成通常晚于触发 step；step 3577 还有同步保存事件，FP8 有 step 0 的保存事件。
因此“间隔保存 4 次/最多保留 4 个”不等于严格只调用保存 4 次。
W&B 耗时仅说明记录了保存动作，无法证明磁盘上的分片齐全或可加载。当前 CPU 节点不可见 GPFS2，checkpoint 完整性仍未验证。

## 7. 异常与数据处理

{table(anomalies) if anomalies else '未发现 CE≤0 的记录。'}

BF16-SiLU 第 894 step 的 CE=0、PPL=1、Z loss=0；梯度范数仍非零，前后 CE 约 3.1。
它与第一次周期 checkpoint 的触发位置重合，但没有证据确定保存就是原因。
原始版不删除此点。另给出全四组共同排除此点的敏感性结果；不会只剔除 SiLU 数据来改变比较样本。
早期 25% 窗口和全程均值受此点影响，末段和验证集比较不受该训练记录影响。
历史中 null/NaN/±Inf 不纳入对应指标的统计；原始 JSONL 保留平台返回的非有限值，需支持 Python JSON 的读取器。

## 8. 证据边界与下一步

这是一组固定种子的 7.5B 受控消融，验证收益的跨数据集一致性是支持证据，{len(evalkeys)} 个验证集不是 {len(evalkeys)} 个独立训练重复。
不能把 I≈0 解释为严格等效，也不能把小的正负 I 解释为显著效应。
训练 loss 与验证 loss 的结论应共同判断；完整 30B 收敛和下游任务表现尚未由这四次运行确定。

若目标是实用结构选择，可以依据本报告训练末段和验证数据判断 Down-RMSNorm 是否值得保留。
若目标是解释长 K 数值机制，应先在已保存权重和同一批输入上比较 Down encoder 的原始输出、
量化后 FP32 参考输出以及 FP8 GEMM 输出，以区分量化误差与累加误差；再比较 Norm 前后方向/尺度误差。
选择性把 Down encoder 改 BF16 的训练实验同时改变量化与累加，提供定位证据但仍非纯累加对照。
预算允许时补多个种子，再评估交互项的可重复性。

## 9. 数据与复现

数据导出时间（UTC）：{meta[LABELS[0]]['exported_at']}。来源 run：

''' + '\n'.join(f'- [{l}]({meta[l]["url"]})' for l in LABELS) + '''

本目录包含每组完整 `*_history.jsonl`、可用表格形式 `*_history.csv`、去除敏感配置键的 `*_metadata.json`，
逐 step 四组合并表 `aligned_training.csv`、阶段统计、效应表、验证集明细、配置审计、异常清单和保存事件。
`overview.pdf` 为矢量图。生成脚本 `../build_full_report.py` 从本地快照生成统计和图，不需要 W&B 登录或 GPU。
旧的 `results_bf16` 保留为历史分析；本报告的统计口径以本节和本目录为准。
'''
    (ROOT/'report.md').write_text(report,encoding='utf-8')
    with zipfile.ZipFile(ROOT/'precision_ablation_report.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(ROOT.iterdir()):
            if path.is_file() and path.suffix!='.zip':
                archive.write(path,path.name)
        for name in ('build_full_report.py','export_four_runs.py'):
            archive.write(ROOT.parent/name, 'scripts/'+name)
    print(table(selected,cols)); print('VALIDATION',lastmacro)
    print('Report:',ROOT/'report.md')


if __name__=='__main__':
    main()
