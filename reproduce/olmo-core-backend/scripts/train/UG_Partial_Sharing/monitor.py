"""Run authorized status/log checks from the local machine every 30 minutes."""

import argparse
import json
import re
import shlex
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
HOST = "cpu-develop.jitao+root.ailab-llmagent.ws@h.pjlab.org.cn"
OUT = Path("/tmp/mose_ug_partial_monitor")
TERMINAL = {"Succeeded", "Stopped", "Failed"}


def registry_paths():
    return [HERE / "jobs.json",
            HERE.parent / "FFN_Rank_Reallocation" / "jobs.json",
            HERE.parent / "Attention_Rank_Reallocation" / "jobs.json",
            HERE.parent / "Rank512_Seed_Replication" / "jobs.json"]


def remote(command):
    result = subprocess.run(
        ["ssh", "-CAXY", HOST, "zsh -lic " + shlex.quote("conda activate MoSE && " + command)],
        capture_output=True, text=True, timeout=120,
    )
    return result.returncode, result.stdout + result.stderr


def check(row, tail_lines=200):
    job = row["job_id"]
    command = (
        f"rjob get --namespace=ailab-llmarchitecture {shlex.quote(job)} && "
        f"rjob logs job --namespace=ailab-llmarchitecture {shlex.quote(job)} --tail-lines {int(tail_lines)}"
    )
    try:
        code, raw = remote(command)
    except subprocess.TimeoutExpired:
        return {**row, "status": "UNKNOWN", "error": "SSH timed out; do not restart",
                "observed_at": datetime.now(ZoneInfo("Asia/Hong_Kong")).isoformat()}
    raw = re.sub(r"wandb_v1_[A-Za-z0-9_-]+", "[REDACTED]", raw)
    raw = re.sub(r"(WANDB_(?:API_KEY|SJY)=)[^\s]+", r"\1[REDACTED]", raw)
    (OUT / f"{row['variant']}.log").write_text(raw)
    status = re.search(r"\(showname=.*?\): (\w+)", raw)
    counts = re.search(r"\d+ replicas (\{[^\n]+\})", raw)
    reports = []
    current = None
    for line in raw.splitlines():
        match = re.search(r"\[step=(\d+)/14306,epoch=\d+,eta=([^\]]+)", line)
        if match:
            current = {"step": int(match[1]), "eta": match[2]}
            reports.append(current)
        if current is not None:
            for key, pattern in (
                ("ce", r"train/CE loss=([\d.eE+-]+)"),
                ("grad", r"optim/total grad norm=([\d.eE+-]+)"),
            ):
                match = re.search(pattern, line)
                if match:
                    current[key] = float(match[1])
    latest = max((x for x in reports if "ce" in x), key=lambda x: x["step"], default=None)
    segments = sorted(set(re.findall(r"/mnt/[^\s'\"\x1b]*offline-run-[^\s'\"\x1b]+", raw)))
    events = [line for line in raw.splitlines() if any(word in line for word in (
        "CUDA out of memory", "SIGTERM", "Training complete", "Traceback (most recent",
        "dp_replicate=", "Loading checkpoint", "Saving checkpoint for step 14306",
    )) or re.search(r"(?:train/CE loss|optim/total grad norm)=(?:nan|[+-]?inf)\b", line, re.I)]
    return {
        **row, "status": status[1] if status else "UNKNOWN", "counts": counts[1] if counts else None,
        "latest": latest, "segments": segments, "events": events[-8:], "observation_exit_code": code,
        "observed_at": datetime.now(ZoneInfo("Asia/Hong_Kong")).isoformat(),
        "observation_basis": "Current scheduler status and job logs",
    }


def check_unfinished(row):
    """Completed jobs need no repeated remote log retrieval."""
    completion = row.get("completion_verified", {})
    if (completion.get("state") == "Succeeded"
            and completion.get("counters", {}).get("active") == 0
            and completion.get("final_step") == 14306
            and completion.get("final_checkpoint_saved") is True):
        return {**row, "status": "Succeeded", "counts": repr(completion["counters"]),
                "latest": None, "segments": [row["wandb_segment"]] if row.get("wandb_segment") else [],
                "events": [], "observation_basis": "Previously verified terminal completion",
                "observed_at": completion.get("observed_at"),
                "terminal_proof": completion.get("proof")}
    return check(row)


def poll():
    started_at = datetime.now(ZoneInfo("Asia/Hong_Kong")).isoformat()
    OUT.mkdir(exist_ok=True)
    jobs = []
    for path in registry_paths():
        if path.exists():
            jobs.extend(json.loads(path.read_text()))
    with ThreadPoolExecutor(max_workers=min(5, len(jobs))) as executor:
        rows = list(executor.map(check_unfinished, jobs))
    stamp = datetime.now(ZoneInfo("Asia/Hong_Kong")).strftime("%Y-%m-%d %H:%M HKT")
    result = {"time": stamp, "started_at": started_at,
              "finished_at": datetime.now(ZoneInfo("Asia/Hong_Kong")).isoformat(), "jobs": rows}
    (OUT / "status.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    with (HERE / "RUN_STATUS.md").open("a") as file:
        file.write(f"\n## {stamp} — monitoring\n\n")
        for row in rows:
            file.write(f"- {row['variant']}: `{row['job_id']}`; {row['status']}; {row.get('counts')}; {row.get('latest')}.\n")
            if row.get("segments"):
                file.write(f"  Exact W&B segments: {row['segments']}.\n")
    summary = {"time": stamp, "started_at": started_at, "finished_at": result["finished_at"], "jobs": [
        {key: row.get(key) for key in ("variant", "status", "counts", "latest", "events", "segments",
                                     "observed_at", "observation_basis", "error")}
        for row in rows
    ]}
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return all(row["status"] in TERMINAL for row in rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--initial-delay", type=float, default=0)
    args = parser.parse_args()
    if args.initial_delay > 0:
        time.sleep(args.initial_delay)
    while True:
        started = time.monotonic()
        finished = poll()
        if finished or not args.loop:
            break
        time.sleep(max(0, 1800 - (time.monotonic() - started)))
