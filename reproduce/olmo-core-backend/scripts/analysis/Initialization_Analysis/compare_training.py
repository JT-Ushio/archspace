"""Strict offline comparison of completed A/C W&B exports (no GPU required).

Input: A_metadata.json, A_history.jsonl, C_metadata.json, C_history.jsonl.
Run only after stopped-attempt synchronization and a complete read-only export.
"""
import argparse
import json
import math
import sys
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "QK_Combination_Analysis"))
from control_trends import NAMING, TRAIN, TOKENS, differences, finite, load_history, write_csv
from plots import chart

LABELS = ("A", "C")
MODES = {"A": "independent", "C": "svd_scale_matched"}


def build(root, *, budget=30_000_000_000, batch_tokens=2_097_152):
    """Validate before writing; preserve the original snapshot including anomalies."""
    meta = {label: json.loads((root / f"{label}_metadata.json").read_text()) for label in LABELS}
    base = meta["A"]["config"]
    audit = []
    for label in LABELS:
        m = meta[label]
        if m["state"] != "finished":
            raise ValueError(f"{label}: not finished")
        if m["config"]["model"].get("factor_init") != MODES[label]:
            raise ValueError(f"{label}: wrong initialization mode")
        if m["config"]["data_loader"]["global_batch_size"] != batch_tokens:
            raise ValueError(f"{label}: global batch mismatch")
        for key, before, after in differences(base, m["config"]):
            audit.append(dict(label=label, field=key, baseline=before, variant=after))
            if key in NAMING:
                continue
            if key != "model.factor_init" or (before, after) != (MODES["A"], MODES["C"]):
                raise ValueError(f"{label}: unexpected configuration difference: {key}")
    end = math.ceil(budget / batch_tokens)
    histories = {label: load_history(root / f"{label}_history.jsonl", batch_tokens) for label in LABELS}
    for label, rows in histories.items():
        if max(rows) != end or meta[label]["summary"]["_step"] != end:
            raise ValueError(f"{label}: incomplete endpoint, expected {end}")
        for step in range(1, end + 1):
            if not finite(rows.get(step, {}).get(TRAIN)):
                raise ValueError(f"{label}: missing/nonfinite train CE at step {step}")

    def is_eval(key):
        return key.startswith("eval/lm/") and key.endswith("/CE loss")

    # Union, not intersection: never silently drop a source/point absent in one run.
    keys = sorted({key for rows in histories.values() for row in rows.values() for key in row if is_eval(key)})
    eval_steps = sorted({step for rows in histories.values() for step, row in rows.items()
                         if any(is_eval(key) for key in row)})
    if not keys or not eval_steps:
        raise ValueError("No validation CE metrics")
    for label, rows in histories.items():
        for step in eval_steps:
            for key in keys:
                if not finite(rows.get(step, {}).get(key)):
                    raise ValueError(f"{label}: missing/nonfinite validation {key} at step {step}")
    last_eval = eval_steps[-1]
    macros = {label: [(step * batch_tokens / 1e9, mean(histories[label][step][key] for key in keys))
                      for step in eval_steps] for label in LABELS}
    final = []
    for label in LABELS:
        row = dict(label=label, initialization=MODES[label], step=end, tokens=end * batch_tokens,
                   validation_step=last_eval, validation_tokens=last_eval * batch_tokens,
                   validation_macro_ce=macros[label][-1][1],
                   validation_gain_vs_A=macros["A"][-1][1] - macros[label][-1][1])
        for window in (100, 500, 1000):
            if end < window:
                raise ValueError("Budget too short for trailing 1000-step comparison")
            row[f"train_ce_last_{window}"] = mean(histories[label][s][TRAIN]
                                                   for s in range(end - window + 1, end + 1))
        final.append(row)
    val_rows = [dict(step=step, tokens=step * batch_tokens, metric=key,
                    A=histories["A"][step][key], C=histories["C"][step][key],
                    gain_A_minus_C=histories["A"][step][key] - histories["C"][step][key])
                for step in eval_steps for key in keys]
    train_rows = [dict(step=step, tokens=step * batch_tokens, A=histories["A"][step][TRAIN],
                      C=histories["C"][step][TRAIN],
                      gain_A_minus_C=histories["A"][step][TRAIN] - histories["C"][step][TRAIN])
                  for step in range(1, end + 1)]
    out = root / "training_comparison"
    out.mkdir(exist_ok=False)
    write_csv(out / "headline.csv", final)
    write_csv(out / "config_differences.csv", audit)
    write_csv(out / "validation_history.csv", val_rows)
    write_csv(out / "validation_final.csv", [r for r in val_rows if r["step"] == last_eval])
    write_csv(out / "train_history.csv", train_rows)
    chart(out / "validation_ce.svg", "Initialization: validation CE (equal-source macro)", macros,
          "Training tokens (B)", "CE (nats / token)")
    chart(out / "validation_gain.svg", "SVD matched gain over independent (positive is better)",
          {"A minus C": [(x, a - c) for (x, a), (_, c) in zip(macros["A"], macros["C"])]},
          "Training tokens (B)", "A CE minus C CE")
    # Nonoverlapping 100-step means; raw every-step CSV remains available.
    series = {label: [(s * batch_tokens / 1e9, mean(histories[label][j][TRAIN]
                       for j in range(s - 99, s + 1))) for s in range(100, end + 1, 100)] for label in LABELS}
    chart(out / "train_ce_100step.svg", "Training CE: nonoverlapping 100-step means", series,
          "Training tokens (B)", "CE (nats / token)")
    report = ["# Initialization A/C training comparison", "",
              "A: independent factors. C: random-dense truncated SVD, encoder/decoder RMS matched separately.",
              f"Training endpoint: {end} steps / {end * batch_tokens / 1e9:.6f}B tokens.",
              f"Last matched validation: {last_eval} steps / {last_eval * batch_tokens / 1e9:.6f}B tokens.",
              "", "| Mode | Validation macro CE | Gain vs A | Train last100 | Last500 | Last1000 |",
              "|---|---:|---:|---:|---:|---:|"]
    report += [f"| {r['label']} | {r['validation_macro_ce']:.6f} | {r['validation_gain_vs_A']:+.6f} | "
               f"{r['train_ce_last_100']:.6f} | {r['train_ce_last_500']:.6f} | {r['train_ce_last_1000']:.6f} |"
               for r in final]
    report += ["", f"Validation macro equally averages {len(keys)} sources; it is not token-weighted CE or BPB.",
               "One paired seed: no across-seed significance claim. Train windows are descriptive, not independent replicates.",
               "The intervention changes factor geometry/spectrum and optimization, not just initialization variance.",
               "SiLU/RMSNorm between factors means UV is not the complete nonlinear mapping.",
               "Checkpoint integrity and GPU initialization verification require the separate artifact/job ledger.", ""]
    report += [f"- {label}: {meta[label].get('url', '')}; exported {meta[label].get('exported_at', '')}."
               for label in LABELS]
    (out / "REPORT.md").write_text("\n".join(report) + "\n")
    return final


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    args = parser.parse_args()
    build(args.snapshot)
