"""Monitor only the five width/gate jobs from the local machine every 30 min."""

import argparse
import ast
import fcntl
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

from capacity import predict
from runtime import HERE, status_helper

HKT = timezone(timedelta(hours=8))


def check(row, tail_lines):
    completion = row.get("completion_verified", {})
    if (completion.get("state") == "Succeeded" and completion.get("final_step") == 14306
            and completion.get("final_checkpoint_saved") is True
            and completion.get("counters", {}).get("active") == 0):
        return {**row, "status": "Succeeded", "counts_dict": completion["counters"],
                "completion_proven": True, "observation_basis": "Previously verified completion"}
    capture_startup = not row.get("startup_capture_saved")
    result = status_helper.check(row, tail_lines=50000 if capture_startup else tail_lines)
    if result.get("counts"):
        result["counts_dict"] = ast.literal_eval(result["counts"])
    log_path = status_helper.OUT / f"{row['variant']}.log"
    raw = log_path.read_text() if log_path.exists() and "error" not in result else ""
    if (result["status"] == "Inqueue" and result.get("counts_dict", {}).get("active") == 0
            and "brainpp/rjob/" in raw and "TypeError: 'NoneType' object is not iterable" in raw):
        # rjob logs fails before a replica has logs. This is a retrieval error,
        # not a traceback emitted by the training process.
        result["log_retrieval_pending"] = True
        result["events"] = []
        raw = ""
    result["initialization_fingerprint"] = row.get("initialization_fingerprint")
    for line in raw.splitlines():
        if "INITIALIZATION_FINGERPRINT " in line:
            result["initialization_fingerprint"] = json.loads(
                line.split("INITIALIZATION_FINGERPRINT ", 1)[1]
            )
    result["training_complete"] = "Training complete" in raw
    final_save_started, final_saved, step0_started, step0_saved = False, False, False, False
    for line in raw.splitlines():
        if "Saving checkpoint for step " in line:
            final_save_started = "Saving checkpoint for step 14306" in line
            step0_started = "Saving checkpoint for step 0 " in line
        if (final_save_started and "Checkpoint saved" in line
                or "Checkpoint for step 14306 saved successfully" in line):
            final_saved = True
        if (step0_started and "Checkpoint saved" in line
                or "Checkpoint for step 0 saved successfully" in line):
            step0_saved = True
    result["completion_proven"] = (
        result["status"] == "Succeeded"
        and result.get("counts_dict", {}).get("active") == 0
        and result.get("counts_dict", {}).get("succeeded") == row["nodes"]
        and result["training_complete"] and final_saved
    )
    startup = row.get("startup_verified", {})
    startup.update({
        "dry_run_complete": startup.get("dry_run_complete", False) or "Dry-run complete" in raw,
        "step0_save_seen": startup.get("step0_save_seen", False) or "Saving checkpoint for step 0 " in raw,
        "step0_checkpoint_saved": startup.get("step0_checkpoint_saved", False) or step0_saved,
    })
    mesh_lines = [line for line in raw.splitlines() if "dp_replicate=" in line]
    if mesh_lines:
        startup["mesh_lines"] = mesh_lines[-2:]
    startup["errors_in_current_tail"] = [line for line in result.get("events", []) if any(
        word in line for word in ("out of memory", "Traceback", "loss=nan", "norm=nan")
    )]
    result["startup_verified"] = startup
    if capture_startup and (result.get("segments") or "Dry-run complete" in raw):
        proof = HERE / f"startup_{row['variant']}.log"
        proof.write_text(raw)
        result["startup_capture_saved"] = str(proof)
    if result["completion_proven"]:
        proof = HERE / f"completion_{row['variant']}.log"
        proof.write_text(raw)
        result["completion_verified"] = {
            "state": "Succeeded", "job_id": row["job_id"],
            "counters": result["counts_dict"], "final_step": 14306,
            "final_checkpoint_saved": True, "observed_at": result["observed_at"],
            "proof": str(proof),
        }
    return result


def poll(tail_lines=1000):
    status_helper.OUT.mkdir(parents=True, exist_ok=True)
    jobs_path = HERE / "jobs.json"
    jobs = json.loads(jobs_path.read_text()) if jobs_path.exists() else []
    active_jobs = [row for row in jobs if row.get("submission_state") not in (
        "retired-terminal", "retired-stopped", "rejected-before-submit"
    )]
    if not active_jobs:
        raise RuntimeError("No confirmed job IDs; do not claim monitoring has started")
    with ThreadPoolExecutor(max_workers=min(5, len(active_jobs))) as executor:
        capacity_futures = [executor.submit(predict, partition, nodes)
                            for partition, nodes in (("arch", 2), ("pool", 1))]
        rows = list(executor.map(lambda row: check(row, tail_lines), active_jobs))
        capacities = []
        for future in capacity_futures:
            try:
                capacities.append(future.result())
            except Exception as error:
                capacities.append({"error": str(error), "schedulable": None})
    stamp = datetime.now(HKT)
    result = {"observed_at": stamp.isoformat(), "interval_seconds": 1800, "jobs": rows,
              "capacity": capacities}
    for row in rows:
        original = next(item for item in jobs if item["job_id"] == row["job_id"])
        for key in ("completion_verified", "startup_verified", "startup_capture_saved",
                    "initialization_fingerprint"):
            if row.get(key):
                original[key] = row[key]
        original["last_observation"] = {key: row.get(key) for key in (
            "status", "counts_dict", "latest", "observed_at", "observation_exit_code", "error"
        )}
        if row.get("segments"):
            original["wandb_segments"] = sorted(set(original.get("wandb_segments", []) + row["segments"]))
    by_variant = {row["variant"]: row for row in rows}
    pair = [by_variant.get(target, {}).get("initialization_fingerprint") for target in ("w1", "w1qk")]
    if all(pair):
        result["gpu_initialization_pair"] = {
            "all_parameter_bytes_match": pair[0] == pair[1], "fingerprints": pair,
        }
    jobs_path.write_text(json.dumps(jobs, ensure_ascii=False, indent=2) + "\n")
    (HERE / "latest_status.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    history = HERE / "monitor_history"
    history.mkdir(exist_ok=True)
    (history / f"status_{stamp.strftime('%Y%m%d_%H%M%S')}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    with (HERE / "RUN_STATUS.md").open("a") as file:
        file.write(f"\n## {stamp.strftime('%Y-%m-%d %H:%M HKT')} — monitoring\n\n")
        for row in rows:
            file.write(f"- {row['variant']}: `{row['job_id']}`; {row['status']}; "
                       f"{row.get('counts_dict')}; {row.get('latest')}.\n")
    print(json.dumps({"observed_at": result["observed_at"], "jobs": [
        {key: row.get(key) for key in ("variant", "job_id", "status", "counts_dict", "latest",
                                     "events", "segments", "initialization_fingerprint", "error",
                                     "completion_proven")} for row in rows
    ], "capacity": capacities, "gpu_initialization_pair": result.get("gpu_initialization_pair")},
        ensure_ascii=False), flush=True)
    return len({row["variant"] for row in rows}) == 5 and all(
        row["status"] in ("Succeeded", "Stopped", "Failed")
        and row.get("counts_dict", {}).get("active") == 0 for row in rows
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--tail-lines", type=int, default=1000)
    parser.add_argument("--initial-delay", type=float, default=0)
    args = parser.parse_args()
    with (HERE / "monitor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.initial_delay > 0:
            time.sleep(args.initial_delay)
        while True:
            start = time.monotonic()
            finished = poll(args.tail_lines)
            if finished or not args.loop:
                break
            time.sleep(max(0, 1800 - (time.monotonic() - start)))
