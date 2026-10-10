"""Predict arch/pool availability without creating a job; run locally."""

import argparse
import json
import re
import shlex
from datetime import datetime, timedelta, timezone

from runtime import HERE, remote


def predict(partition, nodes):
    if (partition, nodes) not in (("arch", 2), ("pool", 1)):
        raise ValueError("Only arch 2x8 and pool 1x8 are used for this experiment")
    source = HERE.parent / "FFN_Rank_Reallocation" / (
        "submit_f3qk.log" if partition == "arch" else "submit_f2qk.log"
    )
    command = next(line[2:] for line in source.read_text().splitlines()
                   if line.startswith("+ rjob submit "))
    args = shlex.split(command)
    args = args[:args.index("--")]
    args[args.index("-P") + 1] = str(nodes)
    args = [f"--name=mose-capacity-width-{partition}-{nodes}x8" if arg.startswith("--name=")
            else arg for arg in args]
    args.extend(["--predict-only=true", "--",
                 "/mnt/shared-storage-gpfs2/intern-pretrain-shared02/shijiayang-p/archspace-MoSE/"
                 "reproduce/olmo-core-backend/scripts/train/FFN_Width_Ablations/run_w1.sh"])
    code, raw = remote(shlex.join(args))
    observed = datetime.now(timezone(timedelta(hours=8)))
    proof = HERE / f"capacity_{partition}_{observed.strftime('%Y%m%d_%H%M%S')}.log"
    proof.write_text(raw)
    summary = [line for line in raw.splitlines() if any(word in line for word in (
        "核心状态", "总副本数量", "可调度数量"))]
    total = re.search(r"总副本数量[：:]\s*(\d+)", raw)
    reported_replicas = int(total[1]) if total else None
    valid = code == 0 and reported_replicas == nodes
    result = {"partition": partition, "nodes": nodes, "exit_code": code,
              "observed_at": observed.isoformat(),
              "valid_prediction": valid, "reported_replicas": reported_replicas,
              "schedulable": any("✅" in line and "核心状态" in line
                                  for line in summary) if valid else None,
              "error": None if valid else "Prediction replica count differs from the request; availability is unknown",
              "summary": summary, "proof": str(proof)}
    (HERE / f"latest_capacity_{partition}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("partition", choices=("arch", "pool"))
    args = parser.parse_args()
    print(json.dumps(predict(args.partition, 2 if args.partition == "arch" else 1),
                     ensure_ascii=False), flush=True)
