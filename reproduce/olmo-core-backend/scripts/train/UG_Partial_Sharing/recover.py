"""Sync and replace verified failed attempts with the same experiment paths."""

import ast
import argparse
import json
import re
import shlex
from datetime import datetime
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo

from monitor import HERE, remote, registry_paths
import submit as submission

SEGMENTS = {
    "p1": "offline-run-20261006_112914-olmo3-1b-ug-partial-p1-s560-p0-linear-down-rms-fp8-30b",
    "p4": "offline-run-20261006_112920-olmo3-1b-ug-partial-p4-s160-p368-linear-down-rms-fp8-30b",
    "f1": "offline-run-20261006_114104-olmo3-1b-ffn-budget-f1-ug384-down768-unshared-linear-rms-fp8-30b",
    "f2": "offline-run-20261006_114106-olmo3-1b-ffn-budget-f2-ug256-down1024-unshared-linear-rms-fp8-30b",
    "f3": "offline-run-20261006_114123-olmo3-1b-ffn-budget-f3-ug480-down672-shared-linear-rms-fp8-30b",
}
WANDB_ROOT = "/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb/"


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2))


def sync(root, old, segment):
    ledger_path = root / "synchronization.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
    entry = ledger.get(old["job_id"])
    if entry is None:
        # P1 was already synchronized manually; verify that exact evidence.
        proof = HERE / "sync_p1_completion.log"
        if old["variant"] == "p1" and segment == WANDB_ROOT + SEGMENTS["p1"] and proof.exists() and (
            "Finished syncing " + segment
        ) in proof.read_text():
            entry = {"segment": segment, "state": "finished", "proof": str(proof)}
        else:
            entry = {"segment": segment, "state": "started", "invoked_at": datetime.now(ZoneInfo("Asia/Hong_Kong")).isoformat()}
            ledger[old["job_id"]] = entry
            dump(ledger_path, ledger)
            code, raw = remote("wandb_sjy " + shlex.quote(segment))
            safe = re.sub(r"wandb_v1_[A-Za-z0-9_-]+", "[REDACTED]", raw)
            (root / f"sync_{old['job_id']}.log").write_text(safe)
            entry["invocation_exit_code"] = code
            if code != 0:
                entry["state"] = "invocation-uncertain"
            dump(ledger_path, ledger)
            if code != 0:
                raise RuntimeError("Sync invocation failed or uncertain; do not retry blindly")
    ledger[old["job_id"]] = entry
    dump(ledger_path, ledger)
    if entry["state"] == "invocation-uncertain":
        raise RuntimeError("Resolve ambiguous sync invocation before restart")
    if entry["state"] == "started":
        code, raw = remote("tail -n 100 /tmp/wandb_sjy.log")
        safe = re.sub(r"wandb_v1_[A-Za-z0-9_-]+", "[REDACTED]", raw)
        proof = root / f"sync_{old['job_id']}_completion.log"
        proof.write_text(safe)
        if code == 0 and ("Finished syncing " + segment) in raw:
            entry["state"] = "finished"
            entry["proof"] = str(proof)
            dump(ledger_path, ledger)
        elif code == 0 and segment in raw.rsplit("Syncing 1 run(s):", 1)[-1] and any(
            "transactionlog:" in line and "EOF" in line
            for line in raw.rsplit("Syncing 1 run(s):", 1)[-1].splitlines()
        ):
            entry["state"] = "failed"
            entry["error"] = "transactionlog: EOF; original segment preserved"
            entry["proof"] = str(proof)
            dump(ledger_path, ledger)
    # The submission skill explicitly permits replacement after launching upload.
    # Keep incomplete or failed uploads visible rather than claiming completion.
    print(old["variant"], "old W&B synchronization:", entry["state"], flush=True)
    return entry


def recover(target, segment=None, partition="pool", nodes=1, script=None):
    if (partition == "pool" and nodes != 1) or (partition == "arch" and nodes not in (1, 2, 4)):
        raise ValueError("Use pool1x8 or an authorized available arch1x8/2x8/4x8 allocation")
    roots = [path.parent for path in registry_paths() if path.exists() and any(
        row["variant"] == target for row in json.loads(path.read_text())
    )]
    if len(roots) != 1:
        raise RuntimeError("Resolve one current registry entry before recovery")
    root = roots[0]
    jobs_path = root / "jobs.json"
    jobs = json.loads(jobs_path.read_text())
    old = next(item for item in jobs if item["variant"] == target)
    code, raw = remote("rjob get --namespace=ailab-llmarchitecture " + shlex.quote(old["job_id"]))
    status = re.search(r"\(showname=.*?\): (\w+)", raw)
    counts = re.search(r"\d+ replicas (\{[^\n]+\})", raw)
    if code != 0 or not status or status[1] not in ("Failed", "Stopped") or not counts:
        raise RuntimeError("Old attempt is not verified terminal")
    counters = ast.literal_eval(counts[1])
    if counters["active"] != 0:
        raise RuntimeError("Old attempt still has active replicas")
    segment = segment or old.get("wandb_segment")
    if not segment and "-r1-" in old["job_id"] and target in SEGMENTS:
        segment = WANDB_ROOT + SEGMENTS[target]
    if not segment:
        raise RuntimeError("Resolve the exact W&B segment of this attempt before recovery")
    synchronization = sync(root, old, segment)
    retired_path = root / "retired_jobs.json"
    retired = json.loads(retired_path.read_text()) if retired_path.exists() else []
    retired.append({**old, "state": status[1], "counters": counters, "segment": segment})
    dump(retired_path, retired)
    dump(jobs_path, [item for item in jobs if item["variant"] != target])
    attempts_path = root / "submission_attempts.json"
    attempts = json.loads(attempts_path.read_text())
    for item in attempts:
        if item.get("job_id") == old["job_id"]:
            item["submission_state"] = "retired-terminal"
            item["terminal_confirmation"] = {"state": status[1], "counters": counters}
    dump(attempts_path, attempts)
    plan_path = root / "submission_plan.json"
    plan = json.loads(plan_path.read_text())
    attempt = re.search(r"-r(\d+)-", old["job_id"])
    retry = int(attempt[1]) + 1 if attempt else 2
    for item in plan:
        if item["variant"] == target:
            item["partition"] = partition
            item["nodes"] = nodes
            if script is not None:
                item["script"] = script
            stamp = datetime.now(ZoneInfo("Asia/Hong_Kong")).strftime("%m%d-%H%M")
            item["job_name"] = f"mose-recover-{target}-r{retry}-p1-{nodes}x8-{partition}-{stamp}-{uuid4().hex[:4]}"
    dump(plan_path, plan)
    with (root / "RUN_STATUS.md").open("a") as file:
        file.write(f"\nRecovery {target}: old `{old['job_id']}` verified {status[1]}/active0. Exact old W&B synchronization state: {synchronization['state']}. Model/run/hyperparameters preserved. Replacement script: {script or old['script']}; allocation {partition}{nodes}x8 priority1.\n")
    submission.HERE = root
    submission.submit(target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("targets", nargs="+", choices=["p0", "p1", "p2", "p3", "p4", "p0qk", "p2qk", "p3qk", "p4qk", "f1", "f2", "f3", "f4", "f5", "f1qk", "f2qk", "f3qk", "f4qk", "f5qk", "a1", "a2", "c0s2", "c2s2", "c6s2", "c0s3", "c2s3", "c6s3"])
    for target in parser.parse_args().targets:
        recover(target)
