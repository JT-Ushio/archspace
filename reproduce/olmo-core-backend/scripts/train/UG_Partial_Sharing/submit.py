"""Submit one authorized, locally validated variant; never retry ambiguity."""

import argparse
import json
import re
import shlex
from datetime import datetime
from zoneinfo import ZoneInfo

from monitor import HERE, remote


def submit(target):
    plan = json.loads((HERE / "submission_plan.json").read_text())
    row = next(item for item in plan if item["variant"] == target)
    jobs_path = HERE / "jobs.json"
    jobs = json.loads(jobs_path.read_text()) if jobs_path.exists() else []
    attempts_path = HERE / "submission_attempts.json"
    attempts = json.loads(attempts_path.read_text()) if attempts_path.exists() else []
    if any(item["variant"] == target and item.get("submission_state") not in (
        "rejected-before-submit", "retired-stopped", "retired-terminal"
    ) for item in jobs + attempts):
        raise RuntimeError("An attempt already exists; inspect its actual status before replacing it")
    # The gateway names the architecture pool partition 'archp'.
    partition_alias = "archp" if row["partition"] == "pool" else row["partition"]
    command = shlex.join([
        "gpu_run", partition_alias, str(row["nodes"]), str(row["gpus_per_node"]),
        row["job_name"], row["script"],
    ])
    attempts.append({**row, "submission_state": "started", "time": datetime.now(ZoneInfo("Asia/Hong_Kong")).isoformat()})
    attempts_path.write_text(json.dumps(attempts, ensure_ascii=False, indent=2))
    code, raw = remote("PRIORITY=1 " + command)
    safe_output = re.sub(r"wandb_v1_[A-Za-z0-9_-]+", "[REDACTED]", raw)
    safe_output = re.sub(r"(WANDB_(?:API_KEY|SJY)=)[^\s]+", r"\1[REDACTED]", safe_output)
    (HERE / f"submit_{target}.log").write_text(safe_output)
    # The scheduler may normalize/truncate a complex requested name.
    # Trust the explicit created identifier, rather than reconstructing it.
    identifiers = set(re.findall(r"created rjob_name:\s*(mose-[a-z0-9-]+)", raw))
    if code != 0 or len(identifiers) != 1:
        attempts[-1]["submission_state"] = "unknown"
        attempts_path.write_text(json.dumps(attempts, ensure_ascii=False, indent=2))
        for line in safe_output.splitlines():
            if "[INFO]" in line or "[ERROR]" in line or "错误：" in line:
                print(line, flush=True)
        raise RuntimeError("Submission response is ambiguous; do not submit another copy")
    row["job_id"] = identifiers.pop()
    jobs.append(row)
    jobs_path.write_text(json.dumps(jobs, ensure_ascii=False, indent=2))
    attempts[-1]["submission_state"] = "confirmed"
    attempts[-1]["job_id"] = row["job_id"]
    attempts_path.write_text(json.dumps(attempts, ensure_ascii=False, indent=2))
    with (HERE / "RUN_STATUS.md").open("a") as file:
        file.write(f"\nSubmitted {target.upper()}: `{row['job_id']}`, {row['partition']} {row['nodes']}x8, priority1; script `{row['script']}`.\n")
    for line in safe_output.splitlines():
        if "[INFO]" in line or "[ERROR]" in line:
            print(line, flush=True)
    print(json.dumps(row, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("target", choices=("p0", "p1", "p2", "p3", "p4", "p0qk", "p2qk", "p3qk", "p4qk"))
    submit(parser.parse_args().target)
