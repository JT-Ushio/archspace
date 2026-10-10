"""Plot two verified C0/C6 seed pairs using complete training and final validation."""

import csv
import hashlib
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRAIN_PATH = HERE.parents[1] / "train/Rank512_Seed_Replication/C0_C6_FULL_TRAIN_CE.json"


def load_rows():
    train_payload = TRAIN_PATH.read_text().rstrip("\n")
    train_jobs = json.loads((HERE.parents[1] / "train/UG_Partial_Sharing/reports/jobs.json").read_text())
    train_job = next(row for row in train_jobs if row["variant"] == "c0-c6-seed-confirmation")
    train_completion = train_job["completion_verified"]
    if not (train_completion["state"] == "Succeeded"
            and train_completion["counters"]["active"] == 0
            and hashlib.sha256(train_payload.encode()).hexdigest() == train_completion["sha256"]):
        raise ValueError("Training payload checksum or completion guard failed")
    train = json.loads(train_payload)
    paired = json.loads((HERE / "SEED_C0_C6_FINAL_CHECKPOINT_LM.json").read_text())
    jobs = json.loads((HERE / "jobs.json").read_text())
    job = next(row for row in jobs if row["job_id"] == paired["evaluation_job_id"])
    completion = job["completion_verified"]
    if not (train["full_30b_histories_complete"] and train["end_step"] == 14306
            and train["common_excluded_steps"] == []
            and completion["state"] == "Succeeded"
            and completion["counters"]["active"] == 0
            and paired["n_pairs"] == 2 and not paired["original_seed12536_included"]):
        raise ValueError("Require two complete paired-seed training/validation results")
    rows = []
    for suffix, seed in (("s2", 12537), ("s3", 12538)):
        pair = []
        for config in ("c0", "c6"):
            case = config + suffix
            payload = (HERE / f"{case.upper()}_FINAL_CHECKPOINT_LM.json").read_text().rstrip("\n")
            if hashlib.sha256(payload.encode()).hexdigest() != completion["outputs"][case]["sha256"]:
                raise ValueError("Validation payload checksum differs from completion ledger")
            validation = json.loads(payload)
            provenance = train["provenance"][case]
            source = validation["source"]
            allocation = {"partition": "pool", "nodes": 1, "gpus_per_node": 8, "priority": 1}
            if not (validation["world_size"] == 8 and validation["checkpoint_step"] == 14306
                    and source["source_job_id"] == provenance["job_id"]
                    and source["init_seed"] == provenance["init_seed"] == seed
                    and source["data_seed"] == provenance["data_seed"] == 34521
                    and source["num_params"] == provenance["full_parameter_count"] == 797050880
                    and source["training_allocation"] == provenance["allocation"] == allocation
                    and train["runs"][case]["clean_tail_n"] == 1000
                    and len(validation["rows"]) == 11):
                raise ValueError("Matched source, seed, parameter or allocation guard failed")
            row = {"case": case, "config": config.upper(), "init_seed": seed,
                   "data_seed": 34521, "num_params": 797050880,
                   "tail1000_train_ce": train["runs"][case]["common_clean_last1000_mean"],
                   "corrected_final_macro_ce": validation["corrected_macro_ce"],
                   "source_job_id": source["source_job_id"]}
            pair.append(row)
            rows.append(row)
        reference = next(p for p in paired["pairs"] if p["init_seed"] == seed)
        if abs(pair[1]["corrected_final_macro_ce"] - pair[0]["corrected_final_macro_ce"]
               - reference["c6_minus_c0"]) > 1e-12:
            raise ValueError("Paired validation summary differs from source payloads")
    return rows


def main():
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter, MaxNLocator

    rows = load_rows()
    output = HERE / "figures"
    output.mkdir(exist_ok=True)
    with (output / "paired_seed_c0_c6.csv").open("w") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.7))
    styles = {12537: ("#2563eb", "o", 8), 12538: ("#c2410c", "s", 5)}
    for ax, metric, title in zip(axes,
            ("tail1000_train_ce", "corrected_final_macro_ce"),
            ("Train CE: final 1,000 steps", "Corrected held-out macro CE @14,306")):
        differences = []
        for seed, (color, marker, size) in styles.items():
            pair = [next(r for r in rows if r["init_seed"] == seed and r["config"] == c)
                    for c in ("C0", "C6")]
            values = [row[metric] for row in pair]
            differences.append(values[1] - values[0])
            ax.plot([0, 1], values, color=color, marker=marker, markersize=size,
                    markerfacecolor="white", markeredgewidth=1.8, linewidth=1.8,
                    label=f"Initialization seed {seed}")
        ax.set_title(f"{title}\nMean paired Δ (C6−C0) = {statistics.mean(differences):.6f}",
                     fontsize=10, pad=12)
        ax.set_xticks([0, 1], ["C0", "C6"])
        ax.set_xlim(-0.16, 1.16)
        ax.margins(y=0.18)
        ax.set_ylabel("Cross-entropy (lower is better)")
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.4f"))
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Paired initialization-seed comparison: C0 vs C6", y=0.97, fontsize=13)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center",
               bbox_to_anchor=(0.5, 0.91), ncol=2, frameon=False, fontsize=9)
    fig.text(0.5, 0.045,
             "30B tokens; 797,050,880 parameters; fixed data seed 34521; all training on pool 1×8.\n"
             "Two additional initialization seeds; original seed 12536 excluded; no significance claim.",
             ha="center", va="bottom", fontsize=8, color="#475569")
    fig.subplots_adjust(left=0.10, right=0.98, top=0.73, bottom=0.22, wspace=0.39)
    for suffix in ("svg", "pdf", "png"):
        fig.savefig(output / f"paired_seed_c0_c6.{suffix}", dpi=220)
    plt.close(fig)
    print(str(output.resolve()))


if __name__ == "__main__":
    main()
