"""Compare exact-step W&B snapshots; never merge summary metrics across times.

Usage: python control_trends.py SNAPSHOT_DIR --labels L Q K V O
The snapshot contains LABEL_metadata.json and LABEL_history.jsonl produced by
the existing read-only W&B exporter. Add QK only after exporting its history.
"""
import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean

from plots import chart

TRAIN = "train/CE loss"
TOKENS = "throughput/total tokens"
NAMING = {"trainer.callbacks.comet.name", "trainer.callbacks.wandb.name",
          "trainer.callbacks.wandb.group", "trainer.save_folder"}


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def differences(a, b, prefix=""):
    if isinstance(a, dict) and isinstance(b, dict):
        result = []
        for key in sorted(a.keys() | b.keys()):
            result.extend(differences(a.get(key), b.get(key), f"{prefix}.{key}".strip(".")))
        return result
    return [] if a == b else [(prefix, a, b)]


def load_history(path, batch_tokens):
    rows = {}
    token_checks = 0
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            step = row.get("_step")
            if not finite(step) or step < 0 or int(step) != step or step in rows:
                raise ValueError(f"Invalid/duplicate step in {path}: {step}")
            if finite(row.get(TOKENS)):
                if row[TOKENS] != step * batch_tokens:
                    raise ValueError(f"Step/token mismatch in {path} at {step}")
                token_checks += 1
            rows[int(step)] = row
    if token_checks == 0:
        raise ValueError(f"No optimizer-step/token corroboration in {path}")
    return rows


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build(root, labels):
    if not labels or labels[0] != "L" or len(set(labels)) != len(labels):
        raise ValueError("Unique labels must start with baseline L")
    meta = {label: json.loads((root / f"{label}_metadata.json").read_text()) for label in labels}
    baseline = meta["L"]["config"]
    batch_tokens = baseline["data_loader"]["global_batch_size"]
    audit = []
    for label in labels:
        if label not in {"L", "Q", "K", "V", "O", "QK"}:
            raise ValueError(f"Unsupported label: {label}")
        if meta[label]["state"] != "finished":
            raise ValueError(f"Run {label} not finished")
        allowed = {f"model.block.sequence_mixer.{c.lower()}_nonlinearity" for c in label if c != "L"}
        actual = set()
        for key, before, after in differences(baseline, meta[label]["config"]):
            audit.append(dict(label=label, field=key, baseline=before, variant=after))
            if key in NAMING:
                continue
            if key not in allowed or (before, after) != ("identity", "silu"):
                raise ValueError(f"Unexpected configuration difference: {label} {key}")
            actual.add(key)
        if actual != allowed:
            raise ValueError(f"Missing activation difference: {label}")
    histories = {label: load_history(root / f"{label}_history.jsonl", batch_tokens) for label in labels}
    common = set.intersection(*(set(rows) for rows in histories.values()))
    if not common or any(max(rows) != max(common) for rows in histories.values()):
        raise ValueError("Unequal training endpoints")
    end = max(common)
    expected = meta["L"]["summary"]["_step"]
    if end != expected or any(meta[label]["summary"]["_step"] != end for label in labels):
        raise ValueError("History incomplete or summary endpoint mismatch")
    keys = sorted({key for row in histories["L"].values() for key, value in row.items()
                   if key.startswith("eval/lm/") and key.endswith("/CE loss") and finite(value)})
    if not keys:
        raise ValueError("No validation CE metrics")
    eval_steps = sorted(step for step in common if all(
        all(finite(histories[label][step].get(key)) for key in keys) for label in labels))
    if not eval_steps:
        raise ValueError("No complete matched validation points")
    macros = {label: [(step * batch_tokens / 1e9, mean(histories[label][step][key] for key in keys))
                     for step in eval_steps] for label in labels}
    last_eval = eval_steps[-1]
    final = []
    for label in labels:
        row = dict(label=label, train_step=end, tokens=end * batch_tokens,
                   validation_step=last_eval, validation_tokens=last_eval * batch_tokens,
                   validation_macro_ce=macros[label][-1][1],
                   validation_gain_vs_L=macros["L"][-1][1] - macros[label][-1][1])
        for window in (100, 500):
            steps = range(end - window + 1, end + 1)
            values = [histories[label].get(step, {}).get(TRAIN) for step in steps]
            if not all(finite(value) for value in values):
                raise ValueError(f"Incomplete trailing {window} steps: {label}")
            row[f"train_ce_last_{window}"] = mean(values)
        final.append(row)
    out = root / "trends"
    out.mkdir(exist_ok=False)
    write_csv(out / "headline.csv", final)
    if audit:
        write_csv(out / "config_differences.csv", audit)
    write_csv(out / "validation_history.csv", [dict(label=label, step=step,
              tokens=step * batch_tokens, metric=key, value=histories[label][step][key])
              for label in labels for step in eval_steps for key in keys])
    write_csv(out / "validation_final.csv", [dict(label=label, metric=key,
              step=last_eval, value=histories[label][last_eval][key],
              gain_vs_L=histories["L"][last_eval][key] - histories[label][last_eval][key])
              for key in keys for label in labels])
    chart(out / "validation_ce.svg", "Matched validation CE (unweighted source macro)", macros,
          "Training tokens (B)", "CE (nats / token)")
    gains = {label: [(x, base_y - y) for (x, y), (_, base_y) in zip(macros[label], macros["L"])]
             for label in labels if label != "L"}
    chart(out / "validation_gain.svg", "Validation gain over all-linear (positive is better)", gains,
          "Training tokens (B)", "Baseline CE minus variant CE")
    report = ["# Single-attention SiLU controls", "",
              f"Matched last validation: step {last_eval}, {last_eval * batch_tokens / 1e9:.6f}B tokens.",
              f"Training endpoint: step {end}, {end * batch_tokens / 1e9:.6f}B tokens.", "",
              "| Variant | Validation macro CE | Gain vs L | Last 100-step train CE | Last 500-step train CE |",
              "|---|---:|---:|---:|---:|"]
    report += [f"| {r['label']} | {r['validation_macro_ce']:.6f} | {r['validation_gain_vs_L']:+.6f} | "
               f"{r['train_ce_last_100']:.6f} | {r['train_ce_last_500']:.6f} |" for r in final]
    report += ["", f"Macro averages {len(keys)} validation sources equally; it is not token-weighted CE or BPB.",
               "Raw exported histories are preserved. Duplicate steps and step/token disagreements fail validation.",
               "Only the named activation and run/output naming differ in exported configurations.",
               "One training seed per condition: observed differences are not evidence of statistical equivalence or a causal mechanism.",
               "Training averages use identical terminal step windows, not each run's best point.",
               "The last validation is not silently relabeled as final-step validation.", "", "## Sources", ""]
    report += [f"- {label}: {meta[label]['url']} (snapshot {meta[label]['exported_at']})" for label in labels]
    (out / "report.md").write_text("\n".join(report) + "\n")
    return final


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--labels", nargs="+", default=["L", "Q", "K", "V", "O"])
    args = parser.parse_args()
    print(json.dumps(build(args.snapshot, args.labels), indent=2))
