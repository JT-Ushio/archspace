"""Plot all six verified seed replications without changing older figures."""

import csv
import hashlib
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_rows():
    comparison = json.loads((HERE / "SEED_C0_C2_C6_COMPARISON.json").read_text())
    jobs = json.loads((HERE / "jobs.json").read_text())
    records = {}
    for job_id in comparison["evaluation_job_ids"]:
        job = next(r for r in jobs if r["job_id"] == job_id)
        done = job["completion_verified"]
        if done["state"] != "Succeeded" or done["counters"]["active"] != 0:
            raise ValueError("Require verified completed analysis jobs")
        records.update(done["outputs"])
    payload = Path(comparison["train_json"]).read_text().rstrip("\n")
    if hashlib.sha256(payload.encode()).hexdigest() != records["all_seed_train_ce"]["sha256"]:
        raise ValueError("Full six-run Train CE checksum mismatch")
    train = json.loads(payload)
    if not train["full_30b_histories_complete"] or train["common_excluded_steps"]:
        raise ValueError("Require the complete six-run comparison without exclusions")
    rows = []
    for suffix, seed in (("s2", 12537), ("s3", 12538)):
        for config in ("c0", "c2", "c6"):
            case = config + suffix
            text = (HERE / f"{case.upper()}_FINAL_CHECKPOINT_LM.json").read_text().rstrip("\n")
            if hashlib.sha256(text.encode()).hexdigest() != records[case]["sha256"]:
                raise ValueError("Final validation checksum mismatch")
            val = json.loads(text)
            p = train["provenance"][case]
            if not (val["source"]["source_job_id"] == p["job_id"]
                    and val["source"]["init_seed"] == p["init_seed"] == seed
                    and val["source"]["data_seed"] == p["data_seed"] == 34521
                    and val["source"]["num_params"] == p["full_parameter_count"] == 797050880
                    and val["checkpoint_step"] == 14306 and val["world_size"] == 8
                    and train["runs"][case]["clean_tail_n"] == 1000):
                raise ValueError("Matched source, seed, budget or final-step guard failed")
            rows.append({"case": case, "config": config.upper(), "init_seed": seed,
                         "data_seed": 34521, "num_params": 797050880,
                         "tail1000_train_ce": train["runs"][case]["common_clean_last1000_mean"],
                         "corrected_final_macro_ce": val["corrected_macro_ce"],
                         "source_job_id": p["job_id"]})
    for config, summary in comparison["summary_by_config"].items():
        for field, expected in (("tail1000_train_ce", summary["mean_tail1000_train_ce"]),
                                ("corrected_final_macro_ce", summary["mean_corrected_final_macro_ce"])):
            mean = statistics.mean(r[field] for r in rows if r["config"] == config.upper())
            if abs(mean - expected) > 1e-12:
                raise ValueError("Source values differ from the paired summary")
    return rows


def main():
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter, MaxNLocator

    rows = load_rows()
    output = HERE / "figures"
    output.mkdir(exist_ok=True)
    with (output / "paired_seed_c0_c2_c6.csv").open("w") as file:
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
            group = [next(r for r in rows if r["init_seed"] == seed and r["config"] == c)
                     for c in ("C0", "C2", "C6")]
            values = [row[metric] for row in group]
            differences.append(values[2] - values[1])
            ax.plot([0, 1, 2], values, color=color, marker=marker, markersize=size,
                    markerfacecolor="white", markeredgewidth=1.8, linewidth=1.8,
                    label=f"Initialization seed {seed}")
        ax.set_title(f"{title}\nMean paired Δ (C6−C2) = {statistics.mean(differences):+.6f}",
                     fontsize=10, pad=12)
        ax.set_xticks([0, 1, 2], ["C0", "C2", "C6"])
        ax.set_xlim(-0.18, 2.18)
        ax.margins(y=0.18)
        ax.set_ylabel("Cross-entropy (lower is better)")
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.4f"))
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Paired initialization-seed comparison: C0, C2 and C6", y=0.97, fontsize=13)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center",
               bbox_to_anchor=(0.5, 0.91), ncol=2, frameon=False, fontsize=9)
    fig.text(0.5, 0.045,
             "30B tokens; 797,050,880 parameters; fixed data seed 34521; all training on pool 1×8.\n"
             "Two additional initialization seeds; original seed 12536 excluded; no significance claim.",
             ha="center", va="bottom", fontsize=8, color="#475569")
    fig.subplots_adjust(left=0.10, right=0.98, top=0.73, bottom=0.22, wspace=0.39)
    for suffix in ("svg", "pdf", "png"):
        fig.savefig(output / f"paired_seed_c0_c2_c6.{suffix}", dpi=220)
    plt.close(fig)
    print(str(output.resolve()))


if __name__ == "__main__":
    main()
