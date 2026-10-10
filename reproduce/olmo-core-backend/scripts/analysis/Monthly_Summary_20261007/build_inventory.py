"""Build a read-only monthly training inventory from saved evidence.

This reads the whitelisted W&B metadata snapshot and existing local results.
It does not authenticate, submit jobs, change checkpoints, or infer final
quality for unfinished runs. Means retain their source comparison's mask.
"""

import csv
import json
import re
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
BACKEND = HERE.parents[2]
TRAIN = BACKEND / "scripts/train"
ANALYSIS = BACKEND / "scripts/analysis"


def load(path):
    return json.loads(path.read_text())


def family(name):
    if "precision-down-" in name:
        return "精度×Down激活"
    if "down-activation-" in name:
        return "Down激活四格"
    if "all-linear-" in name or "all-lowrank-linear-" in name:
        return "低秩SiLU位置"
    if "init-" in name:
        return "SVD初始化"
    if "down-output-subspace-" in name:
        return "Down输出旋转"
    if "densematched-mose-" in name:
        return "Dense预算全投影因子化"
    if re.search(r"r512matched-c\d-seed", name):
        return "C0/C2/C6种子重复"
    if "r512matched-c" in name:
        return "C0–C8组合"
    if "dense-matched-" in name or "dense-nearmatched-" in name:
        return "Dense单点SiLU"
    if "dense-sharing-" in name:
        return "Dense预算共享"
    if "ug-partial-" in name:
        return "UG部分共享"
    if "ffn-budget-" in name:
        return "FFN秩重分配"
    if "attn-budget-" in name:
        return "Attention秩重分配"
    raise ValueError(f"Unclassified run: {name}")


def parameter_count(name):
    if "densematched-mose-" in name or "dense-sharing-s2-" in name:
        return 1484392448
    if "dense-matched-" in name:
        return 1484916736
    if "dense-nearmatched-" in name:
        return 1483868160
    if "dense-sharing-s1-" in name:
        return 1481246720
    if "dense-sharing-s3-" in name:
        return 1480722432
    if "ug-partial-p1-" in name:
        return 794429440
    if "down-output-subspace-dynamic-residual-" in name:
        return 797050880 + 262144 + 512
    if "down-output-subspace-static-" in name or "down-output-subspace-dynamic-" in name:
        return 797050880 + 262144
    return 797050880


def main():
    catalog = load(HERE / "wandb_catalog_20261007T052031Z.json")
    rows = {}
    for record in catalog["runs"]:
        name = record["name"]
        if any(word in name.lower() for word in ("cross-layer", "ckv", "morphnorm", "mla-")):
            raise ValueError("A run outside the report scope entered the inventory")
        config = record["config"]
        rows[record["id"]] = {
            "family": family(name), "run_id": record["id"], "run_name": name,
            "status": "completed" if record["state"] == "finished" else record["state"],
            "last_step": record["summary"].get("_step"),
            "token_budget": config.get("trainer.max_duration.value"),
            "parameters": parameter_count(name), "init_seed": config.get("init_seed"),
            "data_seed": config.get("data_loader.seed"), "wandb_url": record["url"],
            "wandb_created_at_utc": record["created_at"],
            "creation_time_note": "W&B创建/同步时间，不代替实际训练启动时间",
            "metric": "", "primary_train_ce": "", "full_train_ce": "", "final_raw_ce": "",
            "metric_source": "", "metric_policy": "", "scheduler_job_id": "",
            "status_source": "scripts/analysis/Monthly_Summary_20261007/wandb_catalog_20261007T052031Z.json",
        }

    monitor = load(Path("/tmp/mose_ug_partial_monitor/status.json"))
    observations = {row["variant"]: row for row in monitor["jobs"]}
    for folder in ("UG_Partial_Sharing", "FFN_Rank_Reallocation", "Attention_Rank_Reallocation", "Rank512_Seed_Replication"):
        for job in load(TRAIN / folder / "jobs.json"):
            name = job["wandb_segment"].split("offline-run-", 1)[1][16:]
            if name not in rows:
                obs = observations[job["variant"]]
                rows[name] = {
                    "family": family(name), "run_id": name, "run_name": name,
                    "status": obs.get("status", "unknown"),
                    "last_step": obs.get("latest", {}).get("step"),
                    "token_budget": 30000000000, "parameters": 797050880,
                    "init_seed": 12536, "data_seed": 34521,
                    "wandb_url": "https://wandb.ai/archspace/MoSE/runs/" + name,
                    "wandb_created_at_utc": "", "creation_time_note": "尚未同步的活跃离线run",
                    "metric": "", "primary_train_ce": "", "full_train_ce": "", "final_raw_ce": "",
                    "metric_source": "", "metric_policy": "未完成完整训练，不做末期排名",
                    "scheduler_job_id": job["job_id"], "status_source": f"scripts/train/{folder}/jobs.json; monitor@{monitor['time']}",
                }
            else:
                rows[name]["scheduler_job_id"] = job["job_id"]
                rows[name]["status_source"] += f"; scripts/train/{folder}/jobs.json"

    def metric(run_id, value, source, policy, *, full="", final="", label="last1000_train_ce"):
        if run_id not in rows:
            raise ValueError(f"Metric has no monthly run: {run_id}")
        rows[run_id].update(metric=label, primary_train_ce=value, full_train_ce=full,
                            final_raw_ce=final, metric_source=source, metric_policy=policy)

    latest_source = "scripts/train/UG_Partial_Sharing/P3_P4_QK_WITH_REFERENCES_FULL_TRAIN_CE.json"
    latest = load(BACKEND / latest_source)
    for label, result in latest["runs"].items():
        if label.endswith("-original"):
            continue  # The C0–C8 group below preserves its common 999-point mask.
        run_id = Path(result["paths"][-1]).name.removeprefix("run-").removesuffix(".wandb")
        metric(run_id, result["common_clean_last1000_mean"], latest_source,
               "该20配置比较共同排除12516；末窗1000点；P1另属略小预算",
               full=result["common_clean_full_mean"], final=result["final_raw"])

    seed_source = "scripts/train/Rank512_Seed_Replication/ALL_SEED_FULL_TRAIN_CE.json"
    seeds = load(BACKEND / seed_source)
    for result in seeds["runs"].values():
        run_id = Path(result["paths"][-1]).name.removeprefix("run-").removesuffix(".wandb")
        metric(run_id, result["common_clean_last1000_mean"], seed_source,
               "六次重复共同排除集合为空；末窗1000点",
               full=result["common_clean_full_mean"], final=result["final_raw"])

    # These reports hold the common C0–C8 anomaly mask and are retained verbatim.
    for filename in ("C0_C7_RESULTS.md", "C8_RESULTS.md"):
        source = "scripts/train/Rank512Matched_Combinations/" + filename
        for line in (BACKEND / source).read_text().splitlines():
            cells = [x.strip().replace("**", "") for x in line.split("|")[1:-1]]
            if not cells or not re.fullmatch(r"C[0-8]", cells[0]):
                continue
            label = cells[0].lower()
            run_id = next(key for key in rows if f"r512matched-{label}-" in key and "-seed" not in key)
            if filename == "C0_C7_RESULTS.md":
                full, tail = float(cells[2]), float(cells[3])
            else:
                full, tail = float(cells[1]), float(cells[2])
            metric(run_id, tail, source, "C0–C8共同排除12516、14304；末窗999点；报告值6位小数", full=full)

    for folder in ("Dense_Single_SiLU", "Dense_Sharing"):
        source = f"scripts/train/{folder}/train_ce_comparison.json"
        result = load(BACKEND / source)
        for key, value in result["runs"].items():
            key = key.lower()
            if key == "dense":
                continue
            if folder == "Dense_Sharing":
                run_id = next(k for k in rows if f"dense-sharing-{key}-" in k)
            else:
                token = {"q": "q-only", "k": "k-only", "up": "up-only", "gate": "gate-only"}[key]
                run_id = next(k for k in rows if token in k and ("dense-matched-" in k or "dense-nearmatched-" in k))
            metric(run_id, value["tail_mean"] if folder == "Dense_Sharing" else value["last1000"],
                   source, "无排除，完整14306点、末窗1000点",
                   full=value["full_mean"], final=value["final"] if folder == "Dense_Sharing" else value["final_raw"])

    source = "scripts/analysis/Down_Identity_Ablations/results_four_runs/aligned_training.csv"
    with (BACKEND / source).open() as stream:
        data = list(csv.DictReader(stream))
    for label, name in {"S": "silu", "I": "identity", "R": "rms", "SR": "silu-rms-norm"}.items():
        run_id = f"olmo3-1b-down-activation-{name}-fp8-30b"
        tail = [float(row[label]) for row in data if int(row["step"]) >= 13307]
        full = [float(row[label]) for row in data if int(row["step"]) != 10728]
        if len(tail) != 1000 or len(full) != 14305:
            raise ValueError("Down history coverage changed")
        metric(run_id, sum(tail)/len(tail), source, "四格完整均值共同排除10728；末窗1000点",
               full=sum(full)/len(full), final=float(data[-1][label]))

    source = "scripts/analysis/Precision_Ablations/results_four_runs/training_windows.csv"
    with (BACKEND / source).open() as stream:
        for row in csv.DictReader(stream):
            if row["window"] != "last_10pct" or row["policy"] != "raw":
                continue
            precision, act = row["run"].split("_")
            metric(f"olmo3-1b-precision-down-{act}-{precision}", float(row["mean"]), source,
                   "7.5B短日程末10%，不与30B末1000步混用", label="last10pct_train_ce")

    source = "scripts/analysis/QK_Combination_Analysis/results/final-comparison-20260913/trends/training_windows.csv"
    with (BACKEND / source).open() as stream:
        for row in csv.DictReader(stream):
            if row["window"] != "1000":
                continue
            label = row["label"].lower()
            run_id = "olmo3-1b-all-lowrank-linear-fp8-30b" if label == "l" else (
                "olmo3-1b-all-linear-qk-silu-fp8-30b" if label == "qk" else f"olmo3-1b-all-linear-{label}-only-silu-fp8-30b")
            metric(run_id, float(row["mean_ce"]), source, "本SiLU位置组末1000步；Down为Identity")

    for token, value in {"independent": 2.495849, "svd-scale-matched": 2.494134}.items():
        metric(f"olmo3-1b-init-{token}-down-rms-fp8-30b", value,
               "scripts/analysis/Initialization_Analysis/TRAIN_CE_20260913.md", "原SVD两组末1000步，报告值6位小数")
    for token, value in {"identity": 2.496660, "static": 2.497556, "dynamic": 2.502272, "dynamic-residual": 2.495650}.items():
        metric(f"olmo3-1b-down-output-subspace-{token}-g32-a05-fp8-30b", value,
               "scripts/analysis/Down_Output_Subspace_Ablations/RESULTS_20260922.md", "原旋转四组末1000步，Down为Identity；旋转增加参数")
    for token, value in {"f0-unshared": 2.392441, "f1-kv-gateup-shared": 2.397395}.items():
        metric(f"olmo3-1b-densematched-mose-{token}-down-rms-fp8-30b", value,
               "scripts/analysis/ABLATION_BEST_CONFIGS_20260930.md", "Dense预算全投影因子化组末1000步，报告值6位小数")

    result = sorted(rows.values(), key=lambda row: (row["family"], row["run_name"]))
    if len(result) != 65:
        raise ValueError(f"Expected 61 synchronized runs + 4 active runs, got {len(result)}")
    for row in result:
        if row["status"] == "completed" and not row["metric"]:
            raise ValueError(f"A completed run is missing its result: {row['run_id']}")
    summary = {
        "window_hkt": "2026-09-07 through 2026-10-07 snapshot",
        "wandb_snapshot_utc": catalog["snapshot_utc"], "scheduler_snapshot_hkt": monitor["time"],
        "count_unit": "Distinct logical training run IDs; retries merged, additional seeds separate",
        "counted_training_runs": len(result), "by_family": dict(Counter(row["family"] for row in result)),
        "by_status": dict(Counter(row["status"] for row in result)),
        "additional_unconfirmed_record": "FFN-linear/Attention-SiLU at8890, local README only; not counted as a complete or fully provenance-verified run",
        "runs": result,
    }
    (HERE / "monthly_training_inventory.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    with (HERE / "monthly_training_inventory.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result[0]))
        writer.writeheader()
        writer.writerows(result)
    print(json.dumps({k: summary[k] for k in ("counted_training_runs", "by_family", "by_status", "scheduler_snapshot_hkt")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
