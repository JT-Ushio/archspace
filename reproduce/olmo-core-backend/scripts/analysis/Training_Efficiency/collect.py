"""Collect one efficiency attempt, requiring separate SHA256 for all three cases."""

import ast
import json
import math
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "train/UG_Partial_Sharing"))
import monitor
from recover_report_payload import recover_json

EXPECTED_PARAMS = {"dense": 1_484_916_736, "c0": 797_050_880, "c6": 797_050_880}


def validate(data, case):
    times = data["max_rank_step_seconds"]
    if (data["case"] != case or data["num_params"] != EXPECTED_PARAMS[case]
            or data["world_size"] != 8 or data["global_tokens_per_step"] != 2_097_152
            or data["sequence_length"] != 8192 or data["microbatch_tokens_per_rank"] != 32768
            or data["microbatches_per_rank"] != 8 or data["warmup_steps_excluded"] < 5
            or data["measured_steps"] != 10 or len(times) != 10
            or not data["finite_ce_and_gradients"]
            or not all(math.isfinite(t) and t > 0 for t in times)
            or not math.isclose(data["mean_step_seconds"], statistics.mean(times), rel_tol=1e-12)
            or not math.isclose(data["global_tokens_per_second"],
                                2_097_152 / statistics.mean(times), rel_tol=1e-12)):
        raise ValueError("Reported efficiency recipe, sanity checks or arithmetic differ")


def main():
    jobs = json.loads((HERE / "jobs.json").read_text())
    if len(jobs) != 1:
        raise ValueError("Select one current attempt; do not combine outputs from different jobs")
    row = jobs[0]
    observation = monitor.check(row, tail_lines=10000)
    raw_path = monitor.OUT / (row["variant"] + ".log")
    raw = raw_path.read_text() if raw_path.exists() else ""
    proof = HERE / "completion_observation.log"
    proof.write_text(raw)
    observation["proof"] = str(proof.resolve())
    (HERE / "latest_observation.json").write_text(json.dumps(observation, ensure_ascii=False, indent=2) + "\n")
    outputs, data = {}, {}
    for case in EXPECTED_PARAMS:
        marker = "MOSE_EFFICIENCY_" + case.upper()
        try:
            result = recover_json(raw, marker + "_JSON ", marker + "_SHA256 ")
        except ValueError as error:
            outputs[case] = {"pending": str(error)}
            continue
        validate(result["data"], case)
        data[case] = result["data"]
        path = HERE / (case.upper() + "_STEP_EFFICIENCY.json")
        path.write_text(result["payload"] + "\n")
        outputs[case] = {"sha256": result["sha256"], "path": str(path.resolve()),
                         "mean_step_seconds": data[case]["mean_step_seconds"],
                         "global_tokens_per_second": data[case]["global_tokens_per_second"],
                         "peak_allocated_gib": data[case]["max_rank_peak_allocated_bytes"] / 1024**3}
    counters = ast.literal_eval(observation["counts"]) if observation.get("counts") else {}
    complete = (observation["status"] == "Succeeded" and counters.get("active") == 0
                and counters.get("succeeded") == 1 and len(data) == 3)
    if complete:
        for field in ("gpu", "gpu_total_memory_bytes", "torch_version", "cuda_version", "init_seed"):
            if len({value[field] for value in data.values()}) != 1:
                raise ValueError("Hardware/software/seed differs across the three cases")
        row["completion_verified"] = {"state": observation["status"], "counters": counters,
                                      "observed_at": observation["observed_at"],
                                      "proof": str(proof.resolve()), "outputs": outputs}
        (HERE / "jobs.json").write_text(json.dumps(jobs, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": observation["status"], "counts": counters,
                      "observed_at": observation["observed_at"], "verified_complete": complete,
                      "outputs": outputs, "events": observation.get("events"),
                      "error": observation.get("error")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
