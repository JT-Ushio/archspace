"""Submit one pool8 training variant, preserving the original duplicate guard."""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", choices=("l1", "r0", "r1", "r2", "r3"))
    args = parser.parse_args()
    audit = json.loads((HERE / "CONFIG_AUDIT.json").read_text())
    entry = next(r for r in audit["configs"] if r["variant"] == args.target)
    if entry["parameters"] != 797050880:
        raise RuntimeError("Actual parameter audit failed")
    plan = json.loads((HERE / "submission_plan.json").read_text())
    row = next(r for r in plan if r["variant"] == args.target)
    if not row["submission_allowed"] or (row["partition"], row["nodes"], row["gpus_per_node"]) != ("pool", 1, 8):
        raise RuntimeError("Only an authorized pool1x8 training submission is allowed")
    monitor = load("ffn_residual_submit_remote", HERE.parent / "UG_Partial_Sharing/monitor.py")
    sys.path.insert(0, str(HERE.parent / "UG_Partial_Sharing"))
    helper = load("ffn_residual_submit_shared", HERE.parent / "UG_Partial_Sharing/submit.py")
    helper.HERE, helper.remote = HERE, monitor.remote
    helper.submit(args.target)


if __name__ == "__main__":
    main()
