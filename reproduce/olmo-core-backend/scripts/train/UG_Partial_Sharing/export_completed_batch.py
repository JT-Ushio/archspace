"""Export complete Train CE in a scheduled job where the W&B mount is visible."""

import argparse
import json
import hashlib
import statistics
from pathlib import Path

from compare_offline import compare


HERE = Path(__file__).resolve().parent
BATCH = {"p0", "p1", "p2", "p3", "p4", "p0qk", "p2qk", "f1", "f2", "f3"}
PREFIX = "MOSE_COMPLETED_BATCH_JSON "


def export(manifest_path, required_labels=BATCH):
    segments = json.loads(manifest_path.read_text())
    if required_labels is not None and set(segments) != required_labels:
        raise ValueError("The frozen batch must contain exactly the ten approved configs")
    if len(segments) < 2:
        raise ValueError("A completed cohort requires at least two configurations")
    jobs = {}
    for root in (HERE, HERE.parent / "FFN_Rank_Reallocation",
                 HERE.parent / "Attention_Rank_Reallocation", HERE.parent / "Rank512_Seed_Replication"):
        path = root / "jobs.json"
        if not path.exists():
            continue
        for row in json.loads(path.read_text()):
            if row["variant"] in jobs:
                raise ValueError("Duplicate variant in current job registries")
            jobs[row["variant"]] = row
    manifest = {}
    provenance = {}
    for label, entry in segments.items():
        if label not in jobs:
            raise ValueError(f"{label}: no current verified registry entry")
        row = jobs[label]
        if row["job_id"] != entry["job_id"]:
            raise ValueError(f"{label}: attempt changed; refresh the verified segment manifest")
        completion = row.get("completion_verified", {})
        if not (
            completion.get("state") == "Succeeded"
            and completion.get("counters", {}).get("active") == 0
            and completion.get("final_step") == 14306
            and completion.get("final_checkpoint_saved") is True
        ):
            raise ValueError(f"{label}: final training/checkpoint completion is unverified")
        if entry["segments"][-1] != row["wandb_segment"]:
            raise ValueError(f"{label}: latest segment differs from the verified attempt")
        files = []
        for segment in entry["segments"]:
            candidates = list(Path(segment).glob("run-*.wandb"))
            if len(candidates) != 1:
                raise ValueError(f"{label}: expected one W&B history in {segment}")
            files.append(str(candidates[0]))
        manifest[label] = files
        provenance[label] = {
            "job_id": row["job_id"],
            "segments": entry["segments"],
            "completion_verified": completion,
            "full_parameter_count": row.get("num_params", 794429440 if label == "p1" else 797050880),
            "strictly_parameter_matched": row.get("num_params", 794429440 if label == "p1" else 797050880) == 797050880,
            "resume_verified": row.get("resume_verified"),
            "init_seed": row.get("init_seed", 12536),
            "data_seed": row.get("data_seed", 34521),
            "allocation": {key: row[key] for key in ("partition", "nodes", "gpus_per_node", "priority")},
        }
    result = compare(manifest, 14306)
    if not result["full_30b_histories_complete"]:
        missing = {label: row["missing_steps"] for label, row in result["runs"].items() if row["missing_steps"]}
        raise ValueError(f"Cannot rank incomplete histories: {missing}")
    metric = "common_clean_last1000_mean"
    losses = {label: row[metric] for label, row in result["runs"].items()}
    if any(value is None for value in losses.values()):
        raise ValueError("The matched final-window metric is unavailable")
    result["provenance"] = provenance
    result["primary_metric"] = metric
    result["single_seed_screening"] = not any(label.startswith(("c0s", "c2s", "c6s")) for label in segments)
    result["strict_budget_ranking"] = sorted(
        (label for label in segments if provenance[label]["strictly_parameter_matched"]), key=losses.get)
    result["all_config_ranking"] = sorted(segments, key=losses.get)
    result["qk_effects"] = {
        label: losses[label + "qk"] - losses[label]
        for label in ("p0", "p2", "p3", "p4", "f1", "f2", "f3", "f4", "f5")
        if label in losses and label + "qk" in losses
    }
    result["qk_interaction"] = None
    if {"p0", "p0qk", "p2", "p2qk"} <= losses.keys():
        result["qk_interaction"] = {
            "qk_change_unshared": losses["p0qk"] - losses["p0"],
            "qk_change_partial_shared": losses["p2qk"] - losses["p2"],
            "difference_of_differences": (losses["p2qk"] - losses["p2"]) - (losses["p0qk"] - losses["p0"]),
        }
    pairs = []
    contrasts = {"c2_minus_c0": [], "c6_minus_c2": [], "c6_minus_c0": []}
    for suffix in ("s2", "s3"):
        for first, second in (("c0", "c2"), ("c2", "c6"), ("c0", "c6")):
            a, b = first + suffix, second + suffix
            if a not in losses or b not in losses:
                continue
            for key in ("init_seed", "data_seed", "allocation", "full_parameter_count"):
                if provenance[a][key] != provenance[b][key]:
                    raise ValueError(f"{suffix}: paired replications must match {key}")
            contrast = second + "_minus_" + first
            pair = {"init_seed": provenance[a]["init_seed"], "data_seed": provenance[a]["data_seed"],
                    first: losses[a], second: losses[b], contrast: losses[b] - losses[a]}
            contrasts[contrast].append(pair)
            if contrast == "c6_minus_c0":
                pairs.append(pair)
    differences = [pair["c6_minus_c0"] for pair in pairs]
    result["seed_confirmation"] = {
        "pairs": pairs, "n_pairs": len(pairs),
        "mean_c6_minus_c0": statistics.mean(differences) if differences else None,
        "stdev_c6_minus_c0": statistics.stdev(differences) if len(differences) >= 2 else None,
        "original_seed12536_included": False,
        "scope": "Initialization-seed confirmation with fixed data order; no significance claim",
    }
    result["seed_confirmation"]["contrasts"] = {
        name: {"pairs": values, "n_pairs": len(values),
               "mean_difference": statistics.mean(value[name] for value in values) if values else None,
               "stdev_difference": statistics.stdev(value[name] for value in values) if len(values) >= 2 else None}
        for name, values in contrasts.items()
    }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=HERE / "batch_segments.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cohort", action="store_true", help="Use an explicit completed subset instead of the fixed original ten")
    args = parser.parse_args()
    result = export(args.manifest, required_labels=None if args.cohort else BATCH)
    encoded = json.dumps(result, ensure_ascii=False, allow_nan=False)
    args.output.write_text(encoded)
    print("MOSE_COMPLETED_BATCH_SHA256 " + hashlib.sha256(encoded.encode()).hexdigest(), flush=True)
    print(PREFIX + encoded, flush=True)
