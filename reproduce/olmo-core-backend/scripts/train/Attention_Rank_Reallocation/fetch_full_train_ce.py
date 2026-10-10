"""Compare complete, unsampled W&B Train CE histories at identical steps."""

import argparse
import hashlib
import json
import math
import statistics
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import wandb


RUNS = {
    "A1": "olmo3-1b-attn-budget-a1-r832-ug336-down352-unshared-qk-silu-down-rms-fp8-30b",
    "A2": "olmo3-1b-attn-budget-a2-r832-ug320-down448-shared-qk-silu-down-rms-fp8-30b",
    "C0": "olmo3-1b-r512matched-c0-k-silu-down-rms-fp8-30b",
    "C2": "olmo3-1b-r512matched-c2-k-silu-down-rms-gateup-fp8-30b",
    "C6": "olmo3-1b-r512matched-c6-qk-silu-down-rms-gateup-fp8-30b",
}
FFN_RUNS = {
    "F4": "olmo3-1b-ffn-budget-f4-ug400-down816-shared-linear-rms-fp8-30b",
    "F5": "olmo3-1b-ffn-budget-f5-ug560-down528-shared-linear-rms-fp8-30b",
    **{label: RUNS[label] for label in ("C0", "C2", "C6")},
}
END_STEP = 14306


def fetch(label, name, output_dir):
    api = wandb.Api(timeout=30)
    run = api.run(f"archspace/MoSE/{name}")
    if run.id != name or run.name != name:
        raise ValueError(f"{label}: W&B identity differs from the verified run")
    history = {}
    for row in run.scan_history(
        keys=["_step", "train/CE loss"],
        page_size=1000,
        min_step=1,
        max_step=END_STEP + 1,
    ):
        step = int(row["_step"])
        if step not in range(1, END_STEP + 1):
            continue
        if step in history:
            raise ValueError(f"{label}: duplicate optimizer step {step}")
        value = float(row["train/CE loss"])
        if not math.isfinite(value):
            raise ValueError(f"{label}: nonfinite CE at optimizer step {step}")
        history[step] = value
    if set(history) != set(range(1, END_STEP + 1)):
        missing = sorted(set(range(1, END_STEP + 1)) - history.keys())
        raise ValueError(f"{label}: incomplete history; missing {len(missing)} steps")
    encoded = json.dumps(
        {"run_id": name, "run_name": run.name,
         "history": [{"step": s, "train_ce": history[s]} for s in sorted(history)]},
        ensure_ascii=False, allow_nan=False,
    )
    path = output_dir / f"{label.lower()}_full_history.json"
    path.write_text(encoded + "\n")
    print(f"{label}: verified all {END_STEP} Train CE rows", flush=True)
    return label, history, {
        "run_id": name, "run_name": run.name, "wandb_url": run.url,
        "wandb_state": run.state, "source": "W&B API scan_history; no sampling",
        "history_file": path.name,
        "history_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cohort", choices=("attention", "ffn"), default="attention")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = RUNS if args.cohort == "attention" else FFN_RUNS
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(fetch, label, name, args.output_dir)
                   for label, name in runs.items()]
        fetched = [future.result() for future in futures]
    histories = {label: history for label, history, _ in fetched}
    bad = sorted({step for h in histories.values()
                  for step, value in h.items() if value <= 0})
    clean = [s for s in range(1, END_STEP + 1) if s not in bad]
    tail = [s for s in clean if s >= END_STEP - 999]
    result = {
        "observed_at_hkt": datetime.now(ZoneInfo("Asia/Hong_Kong")).isoformat(),
        "end_step": END_STEP, "full_30b_histories_complete": True,
        "common_excluded_steps": bad, "clean_full_n": len(clean),
        "tail_start": END_STEP - 999, "clean_tail_n": len(tail),
        "primary_metric": "common_clean_last1000_mean", "runs": {},
        "scope": "Single initialization seed12536, data seed34521; descriptive architecture comparison",
    }
    for label, history, provenance in fetched:
        result["runs"][label] = {
            **provenance, "rows_in_window": len(history),
            "nonpositive_steps": [s for s, v in history.items() if v <= 0],
            "raw_full_mean": statistics.mean(history.values()),
            "raw_last1000_mean": statistics.mean(history[s] for s in range(END_STEP - 999, END_STEP + 1)),
            "common_clean_full_mean": statistics.mean(history[s] for s in clean),
            "common_clean_last1000_mean": statistics.mean(history[s] for s in tail),
            "final_raw": history[END_STEP],
        }
    c6 = result["runs"]["C6"][result["primary_metric"]]
    for row in result["runs"].values():
        row["delta_vs_c6"] = row[result["primary_metric"]] - c6
    result["ranking"] = sorted(result["runs"], key=lambda k: result["runs"][k][result["primary_metric"]])
    name = ("A1_A2_WITH_REFERENCES_FULL_TRAIN_CE.json" if args.cohort == "attention"
            else "F4_F5_WITH_REFERENCES_FULL_TRAIN_CE.json")
    path = args.output_dir / name
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(path), "excluded_steps": bad,
                      "clean_tail_n": len(tail), "ranking": result["ranking"],
                      "runs": {k: {metric: r[metric] for metric in (
                          "common_clean_full_mean", "common_clean_last1000_mean", "delta_vs_c6")}
                               for k, r in result["runs"].items()}}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
