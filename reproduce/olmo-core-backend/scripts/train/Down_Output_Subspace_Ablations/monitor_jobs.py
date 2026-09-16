"""Local, read-only SSH monitor for these three authorized jobs. Never submits/restarts.

Run --daemon to survive the interactive session; artifacts stay in monitor-output/.
No credentials are loaded, no remote development is performed. SSH remains strict.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "monitor-output"
HOST = "cpu-develop.jitao+root.ailab-llmagent.ws@h.pjlab.org.cn"
JOBS = {
    "identity": "mose-down-subspace-identity-r2-2x8-0916-39459674",
    "static": "mose-down-subspace-static-r3-2x8-0916-75612222",
    "dynamic": "mose-down-subspace-dynamic-r2-2x8-0916-35036741",
}
TERMINAL = {"Succeeded", "Failed", "Stopped"}
KEY_LINES = re.compile(
    r"CE loss|Training complete|Checkpoint saved|Saving checkpoint|offline-run-|"
    r"STEP|Step|step [0-9]|ERROR|Error:|Traceback|NaN|nan loss|out of memory|"
    r"Applied FSDP|Applied torch.compile|output_subspace|GroupedOutputRotation")


def now():
    return datetime.now(timezone.utc).isoformat()


def remote(args, *, logs=False):
    entry = ("https://h.pjlab.org.cn" if logs else
             "http://wangyixiuan-cpu.linzhouhan.ailab-llmarchitecture.svc.pjlab.local:11451")
    command = "conda activate MoSE && " + shlex.join([
        "env", "-u", "http_proxy", "-u", "https_proxy", "-u", "HTTP_PROXY", "-u", "HTTPS_PROXY",
        f"KUBEBRAIN_CLUSTER_ENTRY={entry}", "KUBEBRAIN_NAMESPACE=ailab-llmarchitecture", "rjob", *args])
    result = subprocess.run([
        "ssh", "-CAXY", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
        "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=2",
        HOST, "zsh -ic " + shlex.quote(command)], text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=90)
    if result.returncode:
        # Avoid persisting startup environments or credential-like diagnostics.
        raise RuntimeError(f"SSH/rjob returned {result.returncode}; state remains unknown")
    return result.stdout


def parse_status(output):
    result = {}
    for label, job in JOBS.items():
        match = re.search(r"rjob " + re.escape(job) + r" .*?: (\w+)\r?\n[^\n]*'active': (\d+)", output)
        if not match:
            raise ValueError(f"Status absent for {label}; not interpreting missing output as completion")
        result[label] = {"job_id": job, "state": match[1], "active": int(match[2])}
    return result


def save(snapshot):
    temp = OUTPUT / "status.tmp"
    temp.write_text(json.dumps(snapshot, indent=2) + "\n")
    temp.replace(OUTPUT / "status.json")
    with (OUTPUT / "history.jsonl").open("a") as f:
        f.write(json.dumps(snapshot) + "\n")


def run(interval):
    OUTPUT.mkdir(exist_ok=True)
    with (OUTPUT / "monitor.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("An existing monitor owns this job set; not starting another")
        (OUTPUT / "pid").write_text(str(os.getpid()) + "\n")
        last_log = {}
        captured_terminal = set()
        while True:
            snapshot = {"checked_at": now(), "pid": os.getpid(), "monitor_state": "active"}
            try:
                jobs = parse_status(remote(["list", *JOBS.values()]))
                snapshot["jobs"] = jobs
                # Save heartbeat immediately, before potentially slower log calls.
                save(snapshot)
                for label, info in jobs.items():
                    terminal = info["state"] in TERMINAL and info["active"] == 0
                    if label in captured_terminal:
                        continue
                    if not (info["active"] > 0 or terminal):
                        continue
                    if not terminal and time.monotonic() - last_log.get(label, -1e9) < 180:
                        continue
                    output = remote(["logs", "job", info["job_id"], "-n", "200"], logs=True)
                    lines = [line for line in output.splitlines() if KEY_LINES.search(line)]
                    lines = [line for line in lines
                             if not re.search(r"API_KEY|WANDB_SJY|password|access_token|secret", line, re.I)]
                    with (OUTPUT / f"{label}.log").open("a") as f:
                        f.write(f"\nCHECK {now()} {info['state']}\n" + "\n".join(lines) + "\n")
                    last_log[label] = time.monotonic()
                    if terminal:
                        captured_terminal.add(label)
                if all(i["state"] in TERMINAL and i["active"] == 0 for i in jobs.values()):
                    snapshot["monitor_state"] = "terminal"
                    snapshot["follow_up"] = "Review terminal logs and synchronize exact W&B segments; no automatic restart."
                    save(snapshot)
                    print(json.dumps(snapshot), flush=True)
                    return
                print(json.dumps(snapshot), flush=True)
            except Exception as exc:
                snapshot.update(monitor_state="query_error", error=str(exc))
                save(snapshot)
                print(json.dumps(snapshot), flush=True)
            time.sleep(interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daemon", action="store_true")
    parser.add_argument("--interval", type=int, default=600)
    opts = parser.parse_args()
    if opts.interval < 30:
        parser.error("interval must be at least 30 seconds")
    if opts.daemon:
        OUTPUT.mkdir(exist_ok=True)
        with (OUTPUT / "monitor.log").open("a") as log:
            process = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()),
                                        "--interval", str(opts.interval)], stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=log, start_new_session=True)
        print(f"Monitor launch PID={process.pid}; verify status.json heartbeat before claiming active")
    else:
        run(opts.interval)
