#!/usr/bin/env python3
"""Collect completed GPU evaluations by exact OLMES alias, without loading weights."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from datetime import datetime
from pathlib import Path

from run_easy_dp_sharded import EASY_SUITES, resolve_suite


def suite_score(name, scores, suites):
    if name not in suites:
        return scores[name]
    config = suites[name]
    if config.get('primary_metric', 'macro') != 'macro':
        raise ValueError(f'Unsupported suite aggregation: {name}')
    return statistics.mean(suite_score(child, scores, suites) for child in config['tasks'])


def gpu_summary(path):
    if not path.exists():
        return {'available': False}
    values = {}
    timestamps = []
    for line in path.read_text().splitlines():
        sample = json.loads(line)
        timestamps.append(sample['time'])
        for index, uuid, utilization, used, total, power in sample['devices']:
            try:
                utilization, used = float(utilization), float(used)
            except ValueError:
                continue
            row = values.setdefault(uuid.strip(), {'index': index, 'utilization': [], 'memory': []})
            row['utilization'].append(utilization)
            row['memory'].append(used)
    flat = [u for row in values.values() for u in row['utilization']]
    if not flat:
        return {'available': False}
    return {
        'available': True,
        'basis': '15-second nvidia-smi samples across entire worker lifetime, including initialization and idle periods',
        'num_gpus': len(values),
        'num_sample_times': len(timestamps),
        'first_sample': timestamps[0],
        'last_sample': timestamps[-1],
        'mean_utilization_percent': statistics.mean(flat),
        'median_utilization_percent': statistics.median(flat),
        'fraction_samples_utilization_ge_90': sum(u >= 90 for u in flat) / len(flat),
        'max_memory_gib': max(m for row in values.values() for m in row['memory']) / 1024,
        'per_gpu': {
            uuid: {
                'index': row['index'],
                'mean_utilization_percent': statistics.mean(row['utilization']),
                'max_memory_gib': max(row['memory']) / 1024,
                'num_samples': len(row['utilization']),
            }
            for uuid, row in values.items()
        },
    }


def collect(job, suites):
    output = Path(job['local_output'])
    complete = json.loads((output / 'evaluation_complete.json').read_text())
    report = json.loads((output / 'execution_report.json').read_text())
    manifest = json.loads((output / 'shard_manifest.json').read_text())
    config = json.loads((output / 'run_config.json').read_text())
    expected = {alias for suite in EASY_SUITES for alias in resolve_suite(suite, suites)}
    scores, primary_metrics, task_hashes, rows = {}, {}, {}, []
    for line in (output / 'metrics-all.jsonl').read_text().splitlines():
        row = json.loads(line)
        alias = row['task_config']['metadata']['alias']
        if alias in scores:
            raise ValueError(f'Duplicate alias: {alias}')
        score = float(row['metrics']['primary_score'])
        if not math.isfinite(score):
            raise ValueError(f'Non-finite score: {alias}')
        scores[alias] = score
        primary_metrics[alias] = row['task_config']['primary_metric']
        task_hashes[alias] = row['task_hash']
        rows.append(row)
    if (
        complete['status'] != 'complete'
        or set(scores) != expected
        or len(scores) != complete['num_tasks']
        or len(scores) != manifest['num_tasks']
        or {row['task_idx'] for row in rows} != set(range(len(scores)))
        or {task['alias'] for task in manifest['tasks']} != expected
    ):
        raise ValueError(f'Incomplete or mismatched result set: {output}')
    if config['model'] != job['model']:
        raise ValueError(f'Model path mismatch: {output}')
    return {
        'variant': job['variant'],
        'job_id': job['job_id'],
        'parameters': job['parameters'],
        'model': job['model'],
        'local_output': str(output),
        'completed_tasks': len(scores),
        'wall_seconds': report['wall_seconds'],
        'workers': report['workers'],
        'run_config': config,
        'suite_scores': {suite: suite_score(suite, scores, suites) for suite in EASY_SUITES},
        'suite_aggregation': 'Recursive equal-weight macro average over the official suite tree; no combined average of accuracy and BPB',
        'scores_by_alias': scores,
        'primary_metric_by_alias': primary_metrics,
        'task_hash_by_alias': task_hashes,
        'gpu': gpu_summary(output / 'gpu_samples.jsonl'),
        'hf_numerical_validation': job.get('hf_numerical_validation', 'not_run'),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ledger', type=Path, default=Path(__file__).with_name('dense_single_silu_eval_jobs.json'))
    parser.add_argument('--olmes-repo', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--baseline', default='dense', help='Variant used for matched task comparisons')
    parser.add_argument('--title', default='Dense与单点SiLU：转换后GPU评测')
    args = parser.parse_args()
    sys.path.insert(0, str(args.olmes_repo))
    from oe_eval.configs.task_suites import TASK_SUITE_CONFIGS

    ledger = json.loads(args.ledger.read_text())
    models = [collect(job, TASK_SUITE_CONFIGS) for job in ledger['jobs']]
    baseline = next(model for model in models if model['variant'] == args.baseline)
    for model in models:
        if model['primary_metric_by_alias'] != baseline['primary_metric_by_alias']:
            raise ValueError('Different primary metrics between models')
        if model['task_hash_by_alias'] != baseline['task_hash_by_alias']:
            raise ValueError('Different evaluation task configurations between models')
        model['suite_delta_vs_baseline'] = {
            suite: model['suite_scores'][suite] - baseline['suite_scores'][suite]
            for suite in EASY_SUITES
        }
        if args.baseline == 'dense':
            model['suite_delta_vs_dense'] = model['suite_delta_vs_baseline']
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        'collected_at': datetime.now().astimezone().isoformat(),
        'ledger': str(args.ledger.resolve()),
        'status': 'complete',
        'baseline_variant': args.baseline,
        'models': models,
    }
    (args.output_dir / 'gpu_results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    with (args.output_dir / 'task_scores.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['alias', 'primary_metric'] + [model['variant'] for model in models])
        for alias in baseline['scores_by_alias']:
            writer.writerow([alias, baseline['primary_metric_by_alias'][alias]] + [model['scores_by_alias'][alias] for model in models])
    lines = [
        '# ' + args.title, '',
        '全部为最终step14306 checkpoint。OLMES Base Easy四套件共192项，BF16、DP8/TP1、每卡batch32；同一离线数据、few-shot及上下文设置。各套件按官方suite树递归等权宏平均，BPB越低越好，RC accuracy越高越好。', '',
        '| 模型 | 参数量 | Code BPB ↓ | Math BPB ↓ | QA RC ↑ | QA BPB ↓ | 实际耗时，分钟 | GPU平均利用率 | 峰值显存，GiB/卡 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for model in models:
        scores = [f"{model['suite_scores'][suite]:.6f}" for suite in EASY_SUITES]
        gpu = model['gpu']
        utilization = f"{gpu['mean_utilization_percent']:.1f}%" if gpu['available'] else 'N/A'
        memory = f"{gpu['max_memory_gib']:.1f}" if gpu['available'] else 'N/A'
        lines.append('| ' + ' | '.join([model['variant'], str(model['parameters']), *scores, f"{model['wall_seconds']/60:.2f}", utilization, memory]) + ' |')
    lines += [
        '', '耗时从启动8个worker到结果汇总结束，包含模型初始化与任务准备，不含容器启动或排队。GPU利用率为每15秒nvidia-smi采样均值，含初始化与空闲时间；并非精确SM Active。', '',
        f'比较基准为{args.baseline}，参数量见表；参数量不同的行不能当作严格同参数对照。代码/数学BPB是参考答案的似然打分，并非生成式HumanEval pass@1或数学解题准确率。', '',
        '本次证明转换后的模型能够完整运行此HF benchmark。原生OLMo与HF数值forward对照尚未运行，不能据此认定二者数值一致。单种子小差异不能直接宣称显著优势。', '',
        '[192项逐任务分数](task_scores.csv)／[完整GPU与worker记录](gpu_results.json)。', '',
        '| 模型 | RJob ID | 完成任务 | 最终worker batch |',
        '| --- | --- | ---: | --- |',
    ]
    for model in models:
        batch_sizes = [worker['batch_size'] for worker in model['workers']]
        lines.append(f"| {model['variant']} | `{model['job_id']}` | {model['completed_tasks']} | {batch_sizes} |")
    (args.output_dir / 'GPU_RESULTS.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'status': 'complete', 'models': len(models), 'output': str(args.output_dir)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
