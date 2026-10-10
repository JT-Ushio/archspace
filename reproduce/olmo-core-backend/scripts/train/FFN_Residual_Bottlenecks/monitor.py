"""Read-only local scheduler/log checks for five residual jobs and C9."""

import argparse
import ast
import fcntl
import importlib.util
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "ffn_residual_monitor_shared", HERE.parent / "UG_Partial_Sharing/monitor.py"
)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
helper.OUT = HERE / "monitor_logs"
helper.OUT.mkdir(exist_ok=True)
TERMINAL = {"Succeeded", "Stopped", "Failed"}


def check(row):
    if row.get("completion_verified"):
        return {**row, "observation_basis": "Previously verified completion"}
    startup = not row.get("startup_capture_saved")
    result = helper.check(row, tail_lines=5000 if startup else 1000)
    if result.get("counts"):
        result["counts_dict"] = ast.literal_eval(result["counts"])
    path = helper.OUT / f"{row['variant']}.log"
    raw = path.read_text() if path.exists() and "error" not in result else ""
    if (result["status"] == "Inqueue" and result.get("counts_dict", {}).get("active") == 0
            and "brainpp/rjob/" in raw and "TypeError: 'NoneType' object is not iterable" in raw):
        result["log_retrieval_pending"] = True
        result["events"] = []
        raw = ""
    reports, current = [], None
    for line in raw.splitlines():
        step = re.search(r"\[step=(\d+)/14306[,\]]", line)
        if step:
            current = {"step": int(step[1])}
            reports.append(current)
        if current is not None:
            ce = re.search(r"train/CE loss=([\d.eE+-]+)", line)
            if ce:
                current["ce"] = float(ce[1])
        if "INITIALIZATION_FINGERPRINT " in line:
            result["initialization_fingerprint"] = json.loads(
                line.split("INITIALIZATION_FINGERPRINT ", 1)[1]
            )
    if reports:
        result["latest_logged_step"] = reports[-1]["step"]
        result["latest"] = next((r for r in reversed(reports) if "ce" in r), None)
    final_started = final_saved = False
    for line in raw.splitlines():
        if "Saving checkpoint for step " in line:
            final_started = "Saving checkpoint for step 14306" in line
        if final_started and "Checkpoint saved" in line:
            final_saved = True
        if "Checkpoint for step 14306 saved successfully" in line:
            final_saved = True
    complete = result["status"] == "Succeeded" and (
        result.get("counts_dict", {}).get("active") == 0
        and result.get("counts_dict", {}).get("succeeded") == row["nodes"]
        and final_saved and "Training complete" in raw
    )
    result["completion_verified"] = complete
    if startup and ("Dry-run complete" in raw or result.get("segments")):
        proof = HERE / f"startup_{row['variant']}.log"
        proof.write_text(raw)
        result["startup_capture_saved"] = str(proof)
    if complete:
        (HERE / f"completion_{row['variant']}.log").write_text(raw)
    return result


def _poll():
    jobs_path = HERE / "jobs.json"
    rows = json.loads(jobs_path.read_text()) if jobs_path.exists() else []
    if not rows:
        raise RuntimeError("No confirmed submissions to monitor")
    c9_path = HERE.parent / "Rank512Matched_Combinations/C9_RUN_STATE.json"
    c9 = json.loads(c9_path.read_text()) if c9_path.exists() else None
    if c9 and c9.get("job_id"):
        rows.append({**c9, "completion_verified": c9.get("completion_verified", False)})
    with ThreadPoolExecutor(max_workers=6) as executor:
        checked = list(executor.map(check, rows))
    stamp = datetime.now(ZoneInfo("Asia/Hong_Kong"))
    result = {"observed_at": stamp.isoformat(), "interval_seconds": 1800, "jobs": checked}
    if c9:
        observation = next((r for r in checked if r["variant"] == "c9"), None)
        if observation:
            for key in ("status", "counts_dict", "observed_at", "completion_verified",
                        "startup_capture_saved", "latest", "latest_logged_step"):
                if key in observation:
                    c9[key] = observation[key]
            if observation.get("latest_logged_step") is not None:
                c9["latest_step"] = observation["latest_logged_step"]
            if observation.get("latest"):
                c9["latest_logged_ce"] = observation["latest"].get("ce")
                c9["latest_logged_ce_step"] = observation["latest"].get("step")
            c9["next_check_at"] = (stamp + timedelta(seconds=1800)).isoformat()
            c9_path.write_text(json.dumps(c9, ensure_ascii=False, indent=2) + "\n")
    compact = [r.get("initialization_fingerprint") for r in checked
               if r["variant"] in ("r0", "r1", "r2", "r3")]
    if len(compact) == 4 and all(compact):
        result["r0_r3_initialization_bytes_identical"] = all(r == compact[0] for r in compact)
    residual_rows = [r for r in checked if r["variant"] != "c9"]
    jobs_path.write_text(json.dumps(residual_rows, ensure_ascii=False, indent=2) + "\n")
    (HERE / "latest_status.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    history = HERE / "monitor_history"
    history.mkdir(exist_ok=True)
    (history / f"{stamp.strftime('%Y%m%d_%H%M%S')}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    summary = {"observed_at": stamp.isoformat(), "jobs": [
        {k: r.get(k) for k in ("variant", "job_id", "status", "counts_dict",
                              "latest_logged_step", "latest", "error", "completion_verified")}
        for r in checked
    ]}
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    with (HERE / "RUN_STATUS.md").open("a") as file:
        file.write(f"\n## {stamp.strftime('%Y-%m-%d %H:%M HKT')} — monitoring\n\n")
        for r in checked:
            file.write(f"- {r['variant']}: {r['status']}; {r.get('counts_dict')}; "
                       f"step={r.get('latest_logged_step')}; {r.get('latest')}.\n")
    return all(r["status"] in TERMINAL and r.get("counts_dict", {}).get("active") == 0
               for r in checked)


def poll():
    # Serialize snapshots while allowing an on-demand check during the loop's sleep.
    with (HERE / "poll.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _poll()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--initial-delay", type=float, default=0)
    args = parser.parse_args()
    if not args.loop:
        poll()
        raise SystemExit(0)
    with (HERE / "monitor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.initial_delay:
            time.sleep(args.initial_delay)
        while True:
            start = time.monotonic()
            terminal = poll()
            if terminal or not args.loop:
                break
            time.sleep(max(0, 1800 - (time.monotonic() - start)))
