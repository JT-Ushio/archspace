"""Keep step14306 in the explicitly authorized MoSE checkpoint root only.

The default is a dry run. --execute removes other numeric step directories only
when the same stage has a readable final checkpoint. No sibling roots, HF
exports, experiment metadata, or symlinks are removed.
"""

import argparse
import fcntl
import json
import os
import re
import shutil
import stat
from datetime import datetime, timezone
from pathlib import Path

from audit_checkpoints import inventory, minimally_complete


ALLOWED_ROOT = Path("/mnt/shared-storage-gpfs2/intern-pretrain-shared02/taoji/ckpts/Olmo3-MoSE")
FINAL = "step14306"
STEP = re.compile(r"step\d+\Z")
STAGE = re.compile(r"stage\d+\Z")


def stamp():
    return datetime.now(timezone.utc).isoformat()


def directory_identity(path):
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode):
        raise ValueError(f"Not a real directory: {path}")
    return [info.st_dev, info.st_ino]


def final_identity(path):
    if not minimally_complete(path):
        raise ValueError(f"No readable final checkpoint: {path}")
    files = [path / "config.json", path / "model_and_optim/.metadata"]
    fingerprint = {"directory": directory_identity(path)}
    fingerprint["metadata"] = [[str(p.relative_to(path)), p.stat().st_size,
                                p.stat().st_mtime_ns] for p in files if p.is_file()]
    return fingerprint


def plan(root):
    if root != ALLOWED_ROOT or root.resolve() != ALLOWED_ROOT:
        raise ValueError(f"Only the literal authorized root is allowed: {ALLOWED_ROOT}")
    root_id = directory_identity(root)
    candidates, retained, skipped = [], [], []
    for run in sorted(root.iterdir()):
        if run.is_symlink() or not run.is_dir():
            continue
        run_id = directory_identity(run)
        for stage in sorted(run.iterdir()):
            if not STAGE.fullmatch(stage.name) or stage.is_symlink() or not stage.is_dir():
                continue
            stage_id = directory_identity(stage)
            steps = sorted(p for p in stage.iterdir() if STEP.fullmatch(p.name) and p.is_dir())
            final = stage / FINAL
            if final.is_symlink() or not minimally_complete(final):
                skipped.append({"run": run.name, "stage": stage.name,
                                "reason": "missing_or_unreadable_final_checkpoint",
                                "steps": [p.name for p in steps]})
                continue
            if any(int(p.name[4:]) > 14306 for p in steps):
                skipped.append({"run": run.name, "stage": stage.name,
                                "reason": "step_beyond_14306_requires_review",
                                "steps": [p.name for p in steps]})
                continue
            final_id = final_identity(final)
            final_details = inventory(final)
            if final_details["errors"] or final_details["symlinks"]:
                skipped.append({"run": run.name, "stage": stage.name,
                                "reason": "final_checkpoint_scan_error_or_symlinks",
                                "steps": [p.name for p in steps]})
                continue
            retained.append({"path": str(final), **final_details})
            for path in steps:
                if path.name == FINAL:
                    continue
                if path.is_symlink():
                    skipped.append({"path": str(path), "reason": "symlink"})
                    continue
                details = inventory(path)
                if details["errors"] or details["symlinks"]:
                    skipped.append({"path": str(path), "reason": "scan_error_or_symlink",
                                    "errors": details["errors"]})
                    continue
                candidates.append({"path": str(path), "final_path": str(final),
                                   "root_identity": root_id, "run_identity": run_id,
                                   "stage_identity": stage_id,
                                   "identity": directory_identity(path),
                                   "final_identity": final_id, **details})
    return {"observed_at": stamp(), "root": str(root), "keep_step": 14306,
            "candidate_count": len(candidates),
            "candidate_logical_bytes": sum(r["logical_bytes"] for r in candidates),
            "candidate_allocated_bytes": sum(r["allocated_bytes"] for r in candidates),
            "retained_count": len(retained), "candidates": candidates,
            "retained": retained, "skipped": skipped}


def validate_candidate(root, row):
    path = Path(row["path"])
    final = Path(row["final_path"])
    if (path.parent != final.parent or path.name == FINAL or not STEP.fullmatch(path.name)
            or path.resolve() != path or not path.is_relative_to(root)
            or path.parent.parent.parent != root):
        raise ValueError(f"Unsafe deletion target: {path}")
    for p, key in ((root, "root_identity"), (path.parent.parent, "run_identity"),
                   (path.parent, "stage_identity"), (path, "identity")):
        if directory_identity(p) != row[key]:
            raise ValueError(f"Directory identity changed: {p}")
    if final.is_symlink() or final_identity(final) != row["final_identity"]:
        raise ValueError(f"Final checkpoint changed: {final}")
    current = inventory(path)
    if current["errors"] or current["symlinks"]:
        raise ValueError(f"Deletion target has errors/symlinks: {path}")
    for key in ("files", "logical_bytes", "allocated_bytes"):
        if current[key] != row[key]:
            raise ValueError(f"Deletion target changed ({key}): {path}")
    return path


def execute_plan(root, report, output):
    if not shutil.rmtree.avoids_symlink_attacks:
        raise RuntimeError("This Python does not provide symlink-safe recursive deletion")
    log_path = output / "deletion_events.jsonl"
    for index, row in enumerate(report["candidates"], 1):
        path = validate_candidate(root, row)
        with log_path.open("a") as log:
            log.write(json.dumps({"at": stamp(), "event": "delete_started",
                                  "path": str(path)}) + "\n")
            log.flush()
            os.fsync(log.fileno())
        shutil.rmtree(path)
        if path.exists() or path.is_symlink():
            raise RuntimeError(f"Deletion not complete: {path}")
        with log_path.open("a") as log:
            log.write(json.dumps({"at": stamp(), "event": "delete_finished",
                                  "path": str(path), "logical_bytes": row["logical_bytes"]}) + "\n")
        print(json.dumps({"deleted": index, "total": report["candidate_count"],
                          "path": str(path)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ALLOWED_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--expected-plan", type=Path)
    args = parser.parse_args()
    if args.root != ALLOWED_ROOT or args.root.resolve() != ALLOWED_ROOT:
        raise ValueError("Root must be the authorized MoSE checkpoint directory")
    if args.output.resolve().is_relative_to(args.root):
        raise ValueError("Audit output must be outside the checkpoint root")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "cleanup.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        report = plan(args.root)
        before = inventory(args.root)
        if before["errors"]:
            raise RuntimeError(f"Root inventory failed: {before['errors']}")
        report["before"] = before
        report["mode"] = "execute" if args.execute else "dry_run"
        if args.expected_plan:
            expected = json.loads(args.expected_plan.read_text())
            def targets(value):
                return [(r["path"], r["identity"], r["final_identity"], r["logical_bytes"])
                        for r in value["candidates"]]
            if targets(expected) != targets(report):
                raise ValueError("Candidates changed since the dry run; review a fresh plan")
        (args.output / "plan.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: report[k] for k in ("mode", "candidate_count",
                            "candidate_logical_bytes", "retained_count", "skipped", "before")}), flush=True)
        if not args.execute:
            return
        execute_plan(args.root, report, args.output)
        after = inventory(args.root)
        final_report = {"finished_at": stamp(), "root": str(args.root),
                        "deleted_count": report["candidate_count"],
                        "deleted_logical_bytes": report["candidate_logical_bytes"],
                        "before": before, "after": after,
                        "retained_count": report["retained_count"], "skipped": report["skipped"],
                        "remaining_nonfinal_numeric_steps": [str(p) for p in args.root.glob("*/stage*/step*")
                            if STEP.fullmatch(p.name) and p.is_dir() and p.name != FINAL]}
        (args.output / "result.json").write_text(json.dumps(final_report, indent=2) + "\n")
        print(json.dumps(final_report), flush=True)
        if after["errors"]:
            raise RuntimeError("Deletion finished, but post-cleanup inventory has errors")


if __name__ == "__main__":
    main()
