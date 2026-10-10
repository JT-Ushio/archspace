"""Compare local W&B histories at common steps, preserving anomalous records.

Manifest format: {"P0": ["/local/run-p0.wandb"], "P2": [...]}.
List resumed segments in chronological order; newer records override old steps.
Use the same script for the FFN-reallocation runs and reference experiments.
"""

import argparse
import json
import math
import statistics
from pathlib import Path

import wandb
from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal.datastore import DataStore
from wandb.sdk.lib import proto_util

wandb._assert_is_internal_process = True


def read(paths):
    train = {}
    validation = {}
    duplicates = []
    for path in paths:
        store = DataStore()
        store.open_for_scan(str(path))
        while True:
            data = store.scan_data()
            if data is None:
                break
            record = wandb_internal_pb2.Record.FromString(data)
            if record.WhichOneof("record_type") != "history":
                continue
            values = proto_util.dict_from_proto_list(record.history.item)
            step = int(values.get("_step", record.history.step.num))
            if "train/CE loss" in values:
                value = float(values["train/CE loss"])
                if step in train:
                    duplicates.append(step)
                train[step] = value
            for key, value in values.items():
                if key.startswith("eval/lm/") and key.endswith("-validation/CE loss"):
                    validation.setdefault(step, {})[key] = float(value)
    return {"train": train, "validation": validation, "duplicate_steps": sorted(set(duplicates))}


def finite_mean(values):
    values = list(values)
    if not values or any(not math.isfinite(x) for x in values):
        return None
    return statistics.mean(values)


def compare(manifest, end_step):
    histories = {label: read(paths) for label, paths in manifest.items()}
    expected = set(range(1, end_step + 1))
    common = expected.intersection(*(set(h["train"]) for h in histories.values()))
    bad = {
        step for h in histories.values() for step, value in h["train"].items()
        if step in expected and (not math.isfinite(value) or value <= 0)
    }
    clean = sorted(common - bad)
    raw = sorted(common)
    tail_start = max(1, end_step - 999)
    clean_tail = [step for step in clean if step >= tail_start]
    raw_tail = [step for step in raw if step >= tail_start]
    result = {
        "end_step": end_step, "tail_start": tail_start,
        "full_30b_histories_complete": end_step == 14306 and common == expected,
        "common_steps": len(common), "common_excluded_steps": sorted(common & bad),
        "raw_statistics_preserve_nonpositive_rows": True,
        "shared_validation_step": None, "shared_validation_sources": [], "runs": {},
    }
    val_steps = set.intersection(*(set(h["validation"]) for h in histories.values()))
    for step in sorted((s for s in val_steps if s <= end_step), reverse=True):
        keys = set.intersection(*(set(h["validation"][step]) for h in histories.values()))
        if len(keys) == 11 and all(
            math.isfinite(h["validation"][step][key])
            for h in histories.values() for key in keys
        ):
            result["shared_validation_step"] = step
            result["shared_validation_sources"] = sorted(keys)
            break
    for label, history in histories.items():
        train = history["train"]
        final = train.get(end_step)
        row = {
            "paths": [str(path) for path in manifest[label]],
            "rows_in_window": len(expected & set(train)),
            "missing_steps": sorted(expected - set(train)),
            "duplicate_steps": history["duplicate_steps"],
            "nonpositive_or_nonfinite_steps": sorted(
                step for step, value in train.items()
                if step in expected and (not math.isfinite(value) or value <= 0)
            ),
            "raw_full_mean": finite_mean(train[s] for s in raw),
            "raw_last1000_mean": finite_mean(train[s] for s in raw_tail),
            "raw_tail_n": len(raw_tail),
            "common_clean_full_mean": finite_mean(train[s] for s in clean),
            "common_clean_last1000_mean": finite_mean(train[s] for s in clean_tail),
            "clean_tail_n": len(clean_tail),
            "final_raw": final if final is None or math.isfinite(final) else None,
            "validation_macro_ce": None,
        }
        step = result["shared_validation_step"]
        if step is not None:
            row["validation_macro_ce"] = statistics.mean(
                history["validation"][step][key] for key in result["shared_validation_sources"]
            )
        result["runs"][label] = row
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--end-step", type=int, default=14306)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if len(manifest) < 2 or args.end_step <= 0:
        parser.error("Provide at least two runs and a positive end step")
    result = compare(manifest, args.end_step)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    summary = {
        "end_step": result["end_step"],
        "complete": result["full_30b_histories_complete"],
        "excluded_steps": result["common_excluded_steps"],
        "validation_step": result["shared_validation_step"],
        "runs": {label: {
            key: row[key] for key in (
                "rows_in_window", "raw_full_mean", "raw_last1000_mean",
                "common_clean_last1000_mean", "final_raw", "validation_macro_ce"
            )
        } for label, row in result["runs"].items()},
    }
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False))
