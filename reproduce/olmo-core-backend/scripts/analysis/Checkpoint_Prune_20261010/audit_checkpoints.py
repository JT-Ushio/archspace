"""Read-only checkpoint retention audit. This tool has no deletion operation."""

import argparse
import csv
import json
import os
import re
import stat
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
STEP = re.compile(r"step(\d+)\Z")
REFERENCE = re.compile(r"/([^/]+)/(stage\d+)/step(\d+)(?:/|$)")
KEY_RUN = re.compile(r"dense-baseline|r512matched-c[026]-|qk-adaptive-g1")


def inventory(directory):
    result = {"logical_bytes": 0, "allocated_bytes": 0, "files": 0,
              "symlinks": 0, "multiply_linked_files": 0, "errors": []}
    seen = set()

    def error(exc):
        result["errors"].append(str(exc))

    for parent, dirs, files in os.walk(directory, followlinks=False, onerror=error):
        for name in [*dirs, *files]:
            path = Path(parent) / name
            try:
                info = path.lstat()
                if stat.S_ISLNK(info.st_mode):
                    result["symlinks"] += 1
                elif stat.S_ISREG(info.st_mode):
                    result["files"] += 1
                    result["logical_bytes"] += info.st_size
                    if info.st_nlink > 1:
                        result["multiply_linked_files"] += 1
                    identity = (info.st_dev, info.st_ino)
                    if identity not in seen:
                        result["allocated_bytes"] += info.st_blocks * 512
                        seen.add(identity)
            except OSError as exc:
                error(exc)
    return result


def minimally_complete(directory):
    """Layout check only; does not deserialize weights or validate numerics."""
    try:
        if not (directory / "config.json").is_file():
            return False
        config = json.loads((directory / "config.json").read_text())
        if not isinstance(config, dict) or "model" not in config:
            return False
        state = directory / "model_and_optim"
        if (state / ".metadata").is_file() and (state / ".metadata").stat().st_size > 0:
            return any(p.is_file() and p.stat().st_size > 0 for p in state.glob("*.distcp"))
        return any((directory / n).is_file() and (directory / n).stat().st_size > 0
                   for n in ["model.pt", "model.safetensors"])
    except (OSError, ValueError):
        return False


def reference_steps(path):
    result = set()
    if not path or not path.exists():
        return result
    with path.open() as file:
        for row in csv.DictReader(file):
            match = REFERENCE.search(row["path"])
            if match:
                result.add((match[1], match[2], int(match[3])))
    return result


def audit(root, completed, references, extra_keep):
    if not root.is_dir():
        raise FileNotFoundError(f"Checkpoint root is not mounted/readable: {root}")
    if root.is_symlink():
        raise ValueError("Use the actual checkpoint root, not a symlink")
    stages = []
    if re.fullmatch(r"stage\d+", root.name):
        stages.append(root)
    else:
        for run in root.iterdir():
            if not run.is_dir() or run.is_symlink():
                continue
            if re.fullmatch(r"stage\d+", run.name):
                stages.append(run)
                continue
            for stage in run.iterdir():
                if re.fullmatch(r"stage\d+", stage.name) and stage.is_dir() and not stage.is_symlink():
                    stages.append(stage)
    if not stages:
        raise ValueError(f"No readable stage directories found under {root}")
    rows = []
    for stage in sorted(stages):
        run = stage.parent.name
        evidence = completed.get(run, {})
        final = evidence.get("final_step")
        final_layout_ok = final is not None and minimally_complete(stage / f"step{final}")
        terminal = evidence.get("state") == "Succeeded" and evidence.get("active") == 0
        eligible = terminal and final_layout_ok and stage.name == "stage1"
        protected = {0, *extra_keep}
        if final is not None:
            protected.add(final)
        if KEY_RUN.search(run):
            protected.add(7152)
        numeric = [(int(m[1]), p) for p in stage.iterdir()
                   if (m := STEP.fullmatch(p.name)) and p.is_dir()]
        if numeric:
            latest = max(number for number, _ in numeric)
            protected.add(latest)
            if final is not None and latest > final:
                eligible = False
        for number, path in sorted(numeric):
            details = inventory(path) if not path.is_symlink() else {
                "logical_bytes": 0, "allocated_bytes": 0, "files": 0,
                "symlinks": 1, "multiply_linked_files": 0, "errors": []}
            complete = minimally_complete(path) if not path.is_symlink() else False
            reasons = []
            if number in protected:
                reasons.append("keep_step0_final_latest_or_key_midpoint")
            if (run, stage.name, number) in references:
                reasons.append("referenced_by_existing_record")
            if not eligible:
                reasons.append("no_verified_terminal_run_or_readable_final_checkpoint")
            if not complete:
                reasons.append("incomplete_or_unsupported_checkpoint_layout")
            if path.is_symlink() or details["symlinks"] or details["errors"]:
                reasons.append("symlink_or_scan_error_requires_manual_review")
            rows.append({"run_name": run, "stage": stage.name, "step": number,
                         "path": str(path), "action": "keep" if reasons else "candidate_only",
                         "reasons": reasons or ["completed_unreferenced_intermediate_checkpoint"],
                         "minimally_complete": complete, **details,
                         "terminal_evidence_observed_at": evidence.get("observed_at")})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--completed-runs", type=Path, default=HERE / "completed_runs.json")
    parser.add_argument("--references-csv", type=Path,
                        default=HERE.parent / "Migration_20261010/absolute_path_references.csv")
    parser.add_argument("--keep-step", type=int, action="append", default=[])
    args = parser.parse_args()
    completed = json.loads(args.completed_runs.read_text()) if args.completed_runs.exists() else {}
    root = args.root.expanduser().absolute()
    rows = audit(root, completed, reference_steps(args.references_csv), set(args.keep_step))
    candidates = [r for r in rows if r["action"] == "candidate_only"]
    report = {"observed_at": datetime.now(timezone.utc).isoformat(), "root": str(args.root),
              "mode": "read_only_no_deletion", "checkpoint_count": len(rows),
              "candidate_count": len(candidates),
              "candidate_logical_bytes": sum(r["logical_bytes"] for r in candidates),
              "candidate_allocated_bytes": sum(r["allocated_bytes"] for r in candidates),
              "note": "Saved terminal evidence and filesystem layout are checked; recheck live jobs before approving deletion. File size is not guaranteed reclaimed space when hard links or snapshots exist.",
              "checkpoints": rows}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "checkpoint_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    fields = ["run_name", "stage", "step", "path", "action", "reasons", "logical_bytes",
              "allocated_bytes", "files", "minimally_complete", "errors"]
    with (args.output / "checkpoint_audit.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({k: v for k, v in report.items() if k != "checkpoints"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
