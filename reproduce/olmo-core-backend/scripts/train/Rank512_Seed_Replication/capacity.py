"""Predict the actual arch/pool resource request without creating a job."""

import json
import shlex
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "UG_Partial_Sharing"))
from monitor import remote


def predict(partition, nodes):
    if partition not in ("arch", "pool") or nodes not in (1, 2, 4):
        raise ValueError("Unsupported prediction")
    source = HERE.parent / "FFN_Rank_Reallocation" / (
        "submit_f3qk.log" if partition == "arch" else "submit_f2qk.log"
    )
    command = next(line[2:] for line in source.read_text().splitlines()
                   if line.startswith("+ rjob submit "))
    args = shlex.split(command)
    boundary = args.index("--")
    args = args[:boundary]
    args[args.index("-P") + 1] = str(nodes)
    args = [f"--name=mose-capacity-{partition}-{nodes}x8" if arg.startswith("--name=")
            else arg for arg in args]
    args.extend(["--predict-only=true", "--", str(HERE / "run_c0s2.sh").replace(
        "/mnt/shared-storage-user/intern-pretrain-shared02/",
        "/mnt/shared-storage-gpfs2/intern-pretrain-shared02/", 1)])
    code, raw = remote(shlex.join(args))
    observed_at = datetime.now(ZoneInfo("Asia/Hong_Kong"))
    proof = HERE / f"capacity_{partition}_{nodes}_{observed_at.strftime('%Y%m%d_%H%M%S')}.log"
    proof.write_text(raw)
    summary = [line for line in raw.splitlines() if any(word in line for word in (
        "核心状态", "总副本数量", "可调度数量"))]
    return {"partition": partition, "nodes": nodes, "exit_code": code,
            "observed_at": observed_at.isoformat(),
            "schedulable": code == 0 and any("✅" in line and "核心状态" in line for line in summary),
            "summary": summary, "proof": str(proof)}


if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=3) as executor:
        predictions = list(executor.map(lambda pair: predict(*pair),
                                       [("arch", 4), ("arch", 2), ("pool", 1)]))
    result = {"time": datetime.now(ZoneInfo("Asia/Hong_Kong")).isoformat(),
              "predictions": predictions}
    (HERE / "latest_capacity.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False), flush=True)
