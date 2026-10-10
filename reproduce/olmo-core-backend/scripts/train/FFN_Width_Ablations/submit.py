"""Submit one explicitly authorized width variant; never retry ambiguity."""

import argparse
import json
import sys
import uuid
from datetime import datetime, timedelta, timezone

from runtime import HERE, load_helper, remote


def submit(target, partition, nodes):
    if (partition, nodes) not in (("arch", 2), ("pool", 1)):
        raise ValueError("Use arch 2x8 or pool 1x8")
    plan_path = HERE / "submission_plan.json"
    plan = json.loads(plan_path.read_text())
    row = next(item for item in plan if item["variant"] == target)
    if row.get("submission_allowed") is not True:
        raise RuntimeError("Submission is held by the user's instruction")
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%m%d-%H%M")
    row.update(partition=partition, nodes=nodes, priority=1,
               job_name=f"mose-width-{target}-p1-{nodes}x8-{stamp}-{uuid.uuid4().hex[:4]}")
    row.setdefault("local_script", row["script"])
    row["script"] = row["remote_script"]
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n")
    sharing = HERE.parent / "UG_Partial_Sharing"
    sys.path.insert(0, str(sharing))
    helper = load_helper("width_shared_submit", sharing / "submit.py")
    helper.HERE, helper.remote = HERE, remote
    helper.submit(target)
    jobs = json.loads((HERE / "jobs.json").read_text())
    confirmed = next(item for item in reversed(jobs) if item["variant"] == target)
    row.update(status="submitted", job_id=confirmed["job_id"])
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", choices=("w1", "w1qk", "w1m", "w2", "g1"))
    parser.add_argument("--partition", choices=("arch", "pool"), required=True)
    parser.add_argument("--nodes", type=int, required=True)
    args = parser.parse_args()
    submit(args.target, args.partition, args.nodes)
