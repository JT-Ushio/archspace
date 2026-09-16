"""Offline final QK comparison with immutable input snapshots and resume audit.

This script does not deduplicate, average, or renumber W&B history. It fails if
the API snapshot contains conflicting steps, tokens, or attempt timestamps.
"""
import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from control_trends import TRAIN, build, finite, load_history, write_csv
from plots import chart

LABELS = ["L", "Q", "K", "V", "O", "QK"]


def audit_resume(rows, restored_step, attempt_started):
    if sorted(rows) != list(range(max(rows) + 1)):
        raise ValueError("Incomplete optimizer-step coverage")
    after = [row for step, row in rows.items() if step > restored_step]
    if not after or any(not finite(row.get("_timestamp")) or
                        row["_timestamp"] < attempt_started for row in after):
        raise ValueError("Post-restore rows contain another attempt or missing timestamps")
    return dict(restored_step=restored_step, continuing_rows=len(after),
                continuing_timestamp_min=min(row["_timestamp"] for row in after),
                continuing_timestamp_max=max(row["_timestamp"] for row in after),
                policy="No deduplication or renumbering; unique API steps and tokens verified; "
                       "all post-restore rows belong to the successful attempt time window.")


def interaction(values):
    # Positive means the joint CE gain exceeds the sum of separate CE gains.
    return values["Q"] + values["K"] - values["L"] - values["QK"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("controls", type=Path)
    parser.add_argument("qk", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--restored-step", type=int, required=True)
    parser.add_argument("--attempt-started", required=True, help="Timezone-aware ISO timestamp")
    args = parser.parse_args()
    started = datetime.fromisoformat(args.attempt_started)
    if started.tzinfo is None:
        raise ValueError("Attempt timestamp must include timezone")
    args.output.mkdir(parents=True, exist_ok=False)
    provenance = []
    for label in LABELS:
        for suffix in ("metadata.json", "history.jsonl"):
            source = (args.qk if label == "QK" else args.controls) / f"{label}_{suffix}"
            target = args.output / source.name
            shutil.copyfile(source, target)
            provenance.append(dict(file=target.name, source=str(source.resolve()),
                                   sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
    meta = json.loads((args.output / "L_metadata.json").read_text())
    batch = meta["config"]["data_loader"]["global_batch_size"]
    histories = {label: load_history(args.output / f"{label}_history.jsonl", batch) for label in LABELS}
    resume = audit_resume(histories["QK"], args.restored_step, started.timestamp())
    (args.output / "provenance.json").write_text(json.dumps(dict(
        generated_at=datetime.now(timezone.utc).isoformat(), files=provenance, resume=resume), indent=2))
    headline = build(args.output, LABELS)
    out = args.output / "trends"
    keys = sorted({key for row in histories["L"].values() for key, value in row.items()
                   if key.startswith("eval/lm/") and key.endswith("/CE loss") and finite(value)})
    steps = [step for step in sorted(histories["L"]) if all(
        all(finite(histories[label][step].get(key)) for key in keys) for label in LABELS)]
    effects = []
    for step in steps:
        values = {label: mean(histories[label][step][key] for key in keys) for label in LABELS}
        effects.append(dict(step=step, tokens=step * batch, **values,
                            QK_minus_L=values["QK"] - values["L"],
                            QK_minus_Q=values["QK"] - values["Q"],
                            QK_minus_K=values["QK"] - values["K"],
                            interaction_gain=interaction(values)))
    write_csv(out / "interaction_history.csv", effects)
    chart(out / "interaction.svg", "QK interaction gain (positive exceeds additive gains)",
          {"QK": [(row["tokens"] / 1e9, row["interaction_gain"]) for row in effects]},
          "Training tokens (B)", "Q + K - L - QK: CE (nats / token)")
    chart(out / "qk_validation_delta.svg", "QK validation CE minus each control (negative is better)",
          {label: [(row["tokens"] / 1e9, row["QK"] - row[label]) for row in effects]
           for label in ("L", "Q", "K")}, "Training tokens (B)", "QK CE minus control CE")
    end = max(histories["L"])
    windows = []
    for label in LABELS:
        for window in (100, 500, 1000):
            vals = [histories[label][step].get(TRAIN) for step in range(end - window + 1, end + 1)]
            if not all(finite(value) for value in vals):
                raise ValueError(f"Missing training CE: {label} window {window}")
            windows.append(dict(label=label, window=window, end_step=end, mean_ce=mean(vals)))
    write_csv(out / "training_windows.csv", windows)
    chart(out / "train_100step_blocks.svg", "Training CE: disjoint 100-step block means", {
        label: [(stop * batch / 1e9, mean(histories[label][s][TRAIN] for s in range(stop - 99, stop + 1)))
                for stop in range(100, end + 1, 100)] for label in LABELS},
        "Training tokens (B)", "CE (nats / token)")
    # Keep every downstream metric separate: BPB/CE/accuracy and v2 are not interchangeable.
    downstream = []
    dkeys = sorted({key for row in histories["L"].values() for key, val in row.items()
                    if key.startswith("eval/downstream/") and finite(val)})
    for key in dkeys:
        matched = [step for step in histories["L"] if all(
            finite(histories[label][step].get(key)) for label in LABELS)]
        if matched:
            step = max(matched)
            downstream.append(dict(metric=key, step=step, tokens=step * batch,
                                   **{label: histories[label][step][key] for label in LABELS}))
    if downstream:
        write_csv(out / "downstream_last_matched.csv", downstream)
    summary = dict(headline=headline, final_interaction=effects[-1],
                   matched_validation_points=len(steps),
                   qk_lower_macro_points={label: sum(row["QK"] < row[label] for row in effects)
                                          for label in LABELS if label != "QK"},
                   qk_better_final_sources={label: sum(histories["QK"][steps[-1]][key] <
                       histories[label][steps[-1]][key] for key in keys) for label in LABELS if label != "QK"},
                   downstream_metrics=len(downstream), resume=resume)
    (out / "comparison_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
