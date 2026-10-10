"""Read-only post-cleanup checks for the authorized final-only execution."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from keep_final_only import ALLOWED_ROOT, FINAL, STEP, final_identity, minimally_complete


def main():
    output = Path(__file__).resolve().parent / "final_only_execution"
    plan = json.loads((output / "plan.json").read_text())
    result = json.loads((output / "result.json").read_text())
    assert Path(plan["root"]) == ALLOWED_ROOT
    assert Path(result["root"]) == ALLOWED_ROOT
    assert ALLOWED_ROOT.resolve() == ALLOWED_ROOT
    events = [json.loads(line) for line in (output / "deletion_events.jsonl").read_text().splitlines()]
    deleted = [e["path"] for e in events if e["event"] == "delete_finished"]
    expected = {r["path"] for r in plan["candidates"]}
    assert len(deleted) == len(set(deleted)) == len(expected)
    assert set(deleted) == expected
    assert all(not Path(p).exists() and not Path(p).is_symlink() for p in expected)
    fingerprints = {r["final_path"]: r["final_identity"] for r in plan["candidates"]}
    checked = []
    for row in plan["retained"]:
        path = Path(row["path"])
        assert path.is_relative_to(ALLOWED_ROOT) and path.name == FINAL
        assert not path.is_symlink() and minimally_complete(path), str(path)
        if str(path) in fingerprints:
            assert final_identity(path) == fingerprints[str(path)], str(path)
        siblings = sorted(p.name for p in path.parent.iterdir() if STEP.fullmatch(p.name) and p.is_dir())
        assert siblings == [FINAL], (str(path), siblings)
        checked.append(str(path))
    preserved_exceptions = []
    for row in plan["skipped"]:
        if "steps" not in row:
            continue
        for step in row["steps"]:
            path = ALLOWED_ROOT / row["run"] / row["stage"] / step
            assert path.is_relative_to(ALLOWED_ROOT) and path.is_dir(), str(path)
            preserved_exceptions.append(str(path))
    filesystem = os.statvfs(ALLOWED_ROOT)
    verification = {
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "root": str(ALLOWED_ROOT), "deleted_count_verified": len(deleted),
        "retained_final_count_verified": len(checked),
        "preserved_exception_checkpoint_count": len(preserved_exceptions),
        "before_logical_bytes": result["before"]["logical_bytes"],
        "after_logical_bytes": result["after"]["logical_bytes"],
        "directory_logical_reduction_bytes": result["before"]["logical_bytes"] - result["after"]["logical_bytes"],
        "filesystem_available_bytes": filesystem.f_bavail * filesystem.f_frsize,
        "filesystem_total_bytes": filesystem.f_blocks * filesystem.f_frsize,
        "all_checks_passed": True,
    }
    (output / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    print(json.dumps(verification))


if __name__ == "__main__":
    main()
