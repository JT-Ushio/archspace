#!/usr/bin/env python3
"""Record scheduler observations and console-log evidence without loading weights."""

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path


def parse_logs(text):
    current_step = None
    metrics = None
    proof = []
    completed_at = None
    checkpoint_at = None
    final_path = None
    pending_final_save = False
    errors = []
    for line in text.splitlines():
        stamp = re.search(r"\[(\d{4}-\d\d-\d\dT[^\]]+)\]", line)
        at = stamp.group(1) if stamp else None
        header = re.search(r"\[step=(\d+)/(\d+),epoch=\d+(?:,eta=([^\]]+))?\]", line)
        if header:
            current_step = int(header.group(1))
            if header.group(3):
                metrics = {"step": current_step, "eta": header.group(3), "at": at}
            if current_step == 14306:
                proof.append(line)
        ce = re.search(r"train/CE loss=([0-9.eE+-]+)", line)
        if ce and current_step is not None:
            if metrics is None or metrics["step"] != current_step:
                metrics = {"step": current_step, "at": at}
            metrics["ce"] = float(ce.group(1))
        save = re.search(r"Saving checkpoint for step 14306 to ['\"]([^'\"]+)['\"]", line)
        if save:
            final_path = save.group(1)
            pending_final_save = True
            proof.append(line)
        if pending_final_save and "Checkpoint saved" in line:
            checkpoint_at = at
            pending_final_save = False
            proof.append(line)
        if "Training complete" in line:
            completed_at = at
            proof.append(line)
        if re.search(r"Traceback|CUDA out of memory|No space left on device|SIGTERM|SIGKILL|\b(?:NaN|nan|Inf|inf)\b", line):
            errors.append(line)
    return {
        "latest_step": current_step,
        "latest_metrics": metrics,
        "training_complete": completed_at is not None,
        "completed_at": completed_at,
        "final_checkpoint_saved": checkpoint_at is not None,
        "checkpoint_saved_at": checkpoint_at,
        "final_checkpoint": final_path,
        "proof": proof,
        "errors": errors,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--next-check-at", required=True)
    args = parser.parse_args()
    base = Path(__file__).resolve().parent
    incoming = json.loads(args.snapshot.read_text())
    stamp = datetime.now(timezone.utc).isoformat()
    folder = base / "monitor_snapshots" / args.snapshot.stem
    folder.mkdir(parents=True, exist_ok=True)
    observations = []
    for row in incoming:
        (folder / f"{row['variant']}.log").write_text(row["logs"])
        (folder / f"{row['variant']}_scheduler.log").write_text(row["scheduler_output"])
        parsed = parse_logs(row["logs"])
        counts = row.get("counts_dict") or {}
        observations.append({
            **{k: v for k, v in row.items() if k not in ("logs", "scheduler_output")},
            **parsed,
            "observed_at": stamp,
            "progress_percent": round(100 * parsed["latest_step"] / 14306, 2) if parsed["latest_step"] else None,
            "completion_verified": row["status"] == "Succeeded" and counts.get("active") == 0
            and counts.get("succeeded") == 1 and parsed["latest_step"] == 14306
            and parsed["training_complete"] and parsed["final_checkpoint_saved"],
            "log_file": str((folder / f"{row['variant']}.log").relative_to(base)),
        })
    snapshot = {"observed_at": stamp, "jobs": observations}
    (folder / "snapshot.json").write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n")
    state_path = base / "TRAINING_MONITOR_20261009.json"
    state = json.loads(state_path.read_text())
    state["snapshots"].append(str((folder / "snapshot.json").relative_to(base)))
    by_variant = {row["variant"]: row for row in observations}
    jobs = json.loads((base / "jobs.json").read_text())
    for job in jobs:
        row = by_variant.get(job["variant"])
        if row:
            job.update(status=row["status"], last_observation=row)
            if row["completion_verified"]:
                job["completion_verified"] = row
    (base / "jobs.json").write_text(json.dumps(jobs, ensure_ascii=False, indent=2) + "\n")
    for job in state["jobs"]:
        if job["variant"] in by_variant:
            job.update(by_variant[job["variant"]])
    all_complete = all(j["completion_verified"] for j in state["jobs"])
    state.update(status="complete" if all_complete else "monitoring",
                 last_check_at=stamp, next_check_at=None if all_complete else args.next_check_at)
    if all_complete:
        state["completed_at"] = stamp
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")
    with (base / "RUN_STATUS.md").open("a") as stream:
        stream.write(f"\n## {stamp} — 半小时监控\n\n")
        stream.write("| 任务 | 状态 | 最新step / 14306 | 最近日志CE（对应step） | 日志ETA | 完成证据 |\n")
        stream.write("| --- | --- | ---: | --- | --- | --- |\n")
        for row in observations:
            metrics = row["latest_metrics"] or {}
            stream.write(f"| {row['variant'].upper()} | {row['status']} | {row['latest_step']} | "
                         f"{metrics.get('ce')}（{metrics.get('step')}） | {metrics.get('eta')} | "
                         f"{'最终checkpoint及训练完成已核实' if row['completion_verified'] else '尚未完成'} |\n")
        next_message = "三条训练全部完成，半小时监控结束。" if all_complete else f"下一次检查{args.next_check_at}。"
        stream.write(f"\n本次记录：[snapshot]({(folder / 'snapshot.json').relative_to(base)})；"
                     f"{next_message}日志CE仅作进度记录。\n")
    print(json.dumps(observations, ensure_ascii=False))


if __name__ == "__main__":
    main()
