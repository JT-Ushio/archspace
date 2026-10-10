"""Export completed width/gate histories and compare them with frozen C2/C6."""

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

import wandb
from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal.datastore import DataStore
from wandb.sdk.lib import proto_util

HERE = Path(__file__).resolve().parent
wandb._assert_is_internal_process = True


def read_history(paths):
    train, validation, alphas, duplicates = {}, {}, {}, []
    for path in paths:
        stream = DataStore()
        stream.open_for_scan(str(path))
        try:
            while True:
                payload = stream.scan_data()
                if payload is None:
                    break
                record = wandb_internal_pb2.Record.FromString(payload)
                if record.WhichOneof('record_type') != 'history':
                    continue
                values = proto_util.dict_from_proto_list(record.history.item)
                step = int(values.get('_step', record.history.step.num))
                if 'train/CE loss' in values:
                    if step in train:
                        duplicates.append(step)
                    train[step] = float(values['train/CE loss'])
                for key, value in values.items():
                    if key.startswith('qk_alpha/'):
                        alphas.setdefault(step, {})[key] = float(value)
                    if key.startswith('eval/lm/') and key.endswith('-validation/CE loss'):
                        validation.setdefault(step, {})[key] = float(value)
        finally:
            stream.close()
    return {'train': train, 'validation': validation, 'alphas': alphas, 'duplicates': duplicates}


def finite_mean(values):
    values = list(values)
    return statistics.mean(values) if values and all(math.isfinite(x) for x in values) else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError('Use a fresh analysis output directory')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    jobs = json.loads((HERE / 'jobs.json').read_text())
    if {row['variant'] for row in jobs} != {'w1', 'w1qk', 'w1m', 'w2', 'g1'}:
        raise ValueError('Require the five selected width/gate experiments')
    histories, provenance = {}, {}
    for row in jobs:
        label = row['variant'].upper()
        completion = row['completion_verified']
        if (completion['state'] != 'Succeeded' or completion['final_step'] != 14306
                or completion['final_checkpoint_saved'] is not True
                or completion['counters']['active'] != 0):
            raise ValueError(f'{label}: final completion has not been verified')
        files = []
        for directory in row['wandb_segments']:
            found = list(Path(directory).glob('run-*.wandb'))
            if len(found) != 1:
                raise ValueError(f'{label}: expected one exact history in {directory}')
            files.extend(found)
        histories[label] = read_history(files)
        provenance[label] = {key: row.get(key) for key in (
            'run_name', 'job_id', 'parameters', 'partition', 'nodes', 'gpus_per_node',
            'completion_verified', 'initialization_fingerprint')}
        provenance[label]['history_files'] = [
            {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in files
        ]
    reference_dir = HERE.parent / 'FFN_Rank_Reallocation/analysis'
    reference_stats = json.loads((reference_dir / 'F4_F5_WITH_REFERENCES_FULL_TRAIN_CE.json').read_text())
    for label in ['C2', 'C6']:
        path = reference_dir / (label.lower() + '_full_history.json')
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        source = reference_stats['runs'][label]
        if digest != source['history_sha256']:
            raise ValueError(f'{label}: frozen reference history changed')
        data = json.loads(path.read_text())
        train = {int(row['step']): float(row['train_ce']) for row in data['history']}
        if len(train) != len(data['history']):
            raise ValueError(f'{label}: duplicate reference steps')
        histories[label] = {'train': train, 'validation': {}, 'alphas': {}, 'duplicates': []}
        provenance[label] = {'run_name': data['run_name'], 'parameters': 797050880,
                             'history_file': str(path), 'history_sha256': digest,
                             'source': 'Frozen complete W&B API history; archived completion'}
    expected = set(range(1, 14307))
    for label, history in histories.items():
        if set(history['train']) != expected or history['duplicates']:
            raise ValueError(f'{label}: missing, extra or duplicate training steps')
    excluded = sorted({step for history in histories.values() for step, value in history['train'].items()
                       if not math.isfinite(value) or value <= 0})
    clean = sorted(expected - set(excluded))
    tail = list(range(13307, 14307))
    clean_tail = [step for step in tail if step not in excluded]
    stats = {'end_step': 14306, 'tail_start': 13307, 'complete': True,
             'primary_metric': 'common_clean_last1000_mean', 'common_excluded_steps': excluded,
             'clean_full_n': len(clean), 'clean_tail_n': len(clean_tail), 'runs': {},
             'provenance': provenance, 'validation_scope': 'Inline validation exported separately; final corrected validation requires independent checkpoint evaluation'}
    for label, history in histories.items():
        train = history['train']
        stats['runs'][label] = {
            'parameters': provenance[label]['parameters'], 'rows': len(train),
            'raw_full_mean': finite_mean(train[step] for step in sorted(expected)),
            'raw_last1000_mean': finite_mean(train[step] for step in tail),
            'common_clean_full_mean': finite_mean(train[step] for step in clean),
            'common_clean_last1000_mean': finite_mean(train[step] for step in clean_tail),
            'final_raw': train[14306],
            'nonpositive_or_nonfinite_steps': [step for step, value in train.items() if not math.isfinite(value) or value <= 0],
        }
    for row in stats['runs'].values():
        for ref in ['C2', 'C6']:
            row['delta_vs_' + ref.lower()] = row['common_clean_last1000_mean'] - stats['runs'][ref]['common_clean_last1000_mean']
    losses = {label: row['common_clean_last1000_mean'] for label, row in stats['runs'].items()}
    stats['q_activation_effect_reference'] = losses['C6'] - losses['C2']
    stats['q_activation_effect_w1'] = losses['W1QK'] - losses['W1']
    stats['q_activation_interaction'] = stats['q_activation_effect_w1'] - stats['q_activation_effect_reference']
    stats['strict_budget_ranking'] = sorted([label for label in losses if provenance[label]['parameters'] == 797050880], key=losses.get)
    (args.output_dir / 'FULL_TRAIN_CE.json').write_text(json.dumps(stats, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    with (args.output_dir / 'train_ce_by_step.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['step', 'tokens', 'common_excluded'] + list(histories))
        for step in sorted(expected):
            writer.writerow([step, step * 2097152, step in excluded] + [h['train'][step] for h in histories.values()])
    extra = {label: {'inline_validation': h['validation'], 'qk_alpha': h['alphas']} for label, h in histories.items() if h['validation'] or h['alphas']}
    (args.output_dir / 'INLINE_VALIDATION_AND_ALPHA.json').write_text(json.dumps(extra, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print('MOSE_WIDTH_TRAIN_SHA256 ' + hashlib.sha256((args.output_dir / 'FULL_TRAIN_CE.json').read_bytes()).hexdigest(), flush=True)
    print(json.dumps({'complete': True, 'excluded': excluded, 'tail_n': len(clean_tail), 'runs': stats['runs']}, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
