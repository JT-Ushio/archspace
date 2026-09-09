"""Read-only export of the four 30B Muon Down activation W&B runs."""
import argparse
import json
import os
import re
import shlex
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import wandb

RUNS = {
    "S": "olmo3-1b-down-activation-silu-fp8-30b",
    "I": "olmo3-1b-down-activation-identity-fp8-30b",
    "R": "olmo3-1b-down-activation-rms-fp8-30b",
    "SR": "olmo3-1b-down-activation-silu-rms-norm-fp8-30b",
}


def sanitize(value):
    if isinstance(value, dict):
        return {k: sanitize(v) for k, v in value.items() if not any(
            secret in k.lower() for secret in
            ("api_key", "password", "secret", "credential", "access_token", "environ"))}
    if isinstance(value, list):
        return [sanitize(v) for v in value]
    return value


def export(label, run_id, output):
    run = wandb.Api(timeout=60).run(f"archspace/MoSE/{run_id}")
    metadata = dict(label=label, path=f"archspace/MoSE/{run.id}", url=run.url,
                    name=run.name, state=run.state, created_at=run.created_at,
                    exported_at=datetime.now(timezone.utc).isoformat(),
                    config=sanitize(run.config), summary=sanitize(dict(run.summary)))
    (output / f"{label}_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False, default=str))
    print(label, run.name, run.state, "step", run.summary.get("_step"), flush=True)
    count = 0
    keys = set()
    with (output / f"{label}_history.jsonl").open("w") as stream:
        for row in run.scan_history(page_size=1000):
            stream.write(json.dumps(sanitize(row), ensure_ascii=False, default=str) + "\n")
            keys.update(row)
            count += 1
    print(label, "exported", count, "rows", len(keys), "metrics", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "results_four_runs")
    parser.add_argument("--credential-file", type=Path)
    args = parser.parse_args()
    if not os.environ.get("WANDB_API_KEY") and args.credential_file:
        # Parse only the authorized assignment; do not execute the prepare script.
        match = re.search(r"^\s*(?:export\s+)?WANDB_SJY=(.+)$", args.credential_file.read_text(), re.M)
        if not match:
            raise SystemExit("WANDB_SJY assignment not found")
        value = shlex.split(match.group(1), comments=True)[0]
        if "$" in value or "`" in value:
            raise SystemExit("Credential must be a literal; set WANDB_API_KEY in the environment instead")
        os.environ["WANDB_API_KEY"] = value
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.glob("*_history.jsonl")):
        raise SystemExit("Existing snapshot: select a fresh output directory")
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(export, label, run_id, args.output_dir) for label, run_id in RUNS.items()]
        for future in futures:
            future.result()


if __name__ == "__main__":
    main()
