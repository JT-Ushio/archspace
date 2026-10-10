"""Plot verified matched-budget training and corrected final validation losses."""

import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRAIN = HERE.parents[1] / "train/UG_Partial_Sharing/F4_F5_QK_WITH_REFERENCES_FULL_TRAIN_CE.json"


def main():
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter, MaxNLocator

    train = json.loads(TRAIN.read_text())
    validation = json.loads((HERE / "SEVEN_CONFIG_FINAL_CHECKPOINT_LM.json").read_text())
    if not train["full_30b_histories_complete"] or validation["checkpoint_step"] != 14306:
        raise ValueError("Require complete training and final checkpoint evaluation")
    cases = {"c0": "c0-original", "c2": "c2-original", "c6": "c6-original",
             "p0qk": "p0qk", "f3qk": "f3qk", "f4qk": "f4qk", "f5qk": "f5qk"}
    rows = []
    for case, key in cases.items():
        v = validation["runs"][case]
        if (v["source"]["num_params"] != 797_050_880
                or train["provenance"][key]["job_id"] != v["source"]["source_job_id"]):
            raise ValueError("Training/evaluation source or parameter budget mismatch")
        rows.append({"case": case, "num_params": 797_050_880,
                     "tail1000_train_ce": train["runs"][key]["common_clean_last1000_mean"],
                     "corrected_final_macro_ce": v["corrected_macro_ce"],
                     "source_job_id": v["source"]["source_job_id"]})
    output = HERE / "figures"
    output.mkdir(exist_ok=True)
    with (output / "matched_quality_seven.csv").open("w") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    styles = {"c0": ("#64748b", "o", (9, 6)),
              "c2": ("#15803d", "*", (12, 0)),
              "c6": ("#2563eb", "*", (12, 0)),
              "p0qk": ("#c2410c", "s", (12, -18)),
              "f3qk": ("#7e22ce", "^", (12, 12)),
              "f4qk": ("#be123c", "D", (9, 6)),
              "f5qk": ("#0f766e", "v", (12, -18))}
    for row in rows:
        case = row["case"]
        color, marker, offset = styles[case]
        x, y = row["tail1000_train_ce"], row["corrected_final_macro_ce"]
        ax.scatter(x, y, color=color, marker=marker,
                   s=125 if marker == "*" else 55, zorder=3)
        ax.annotate(case.upper(), (x, y), xytext=offset, textcoords="offset points",
                    color=color, fontsize=10, weight="bold",
                    arrowprops={"arrowstyle": "-", "color": color, "lw": 0.6})
    ax.set_title("Training and validation loss at the same 797M parameter budget", pad=14)
    ax.set_xlabel("Mean Train CE over the final 1,000 steps")
    ax.set_ylabel("Corrected held-out macro CE at step 14,306")
    ax.set_xlim(2.4742, 2.4870)
    ax.set_ylim(2.7321, 2.7470)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(MaxNLocator(5))
        axis.set_major_formatter(FormatStrFormatter("%.4f"))
    ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.5, 0.035,
             "Lower is better on both axes. Single seed; 30B tokens; 11 held-out sources.\n"
             "Training allocations differ. Seed replication is pending; no significance claim.",
             ha="center", va="bottom", fontsize=8, color="#475569")
    fig.subplots_adjust(left=0.14, right=0.97, top=0.88, bottom=0.20)
    for suffix in ("svg", "pdf", "png"):
        fig.savefig(output / ("train_vs_corrected_validation_seven." + suffix), dpi=200)
    plt.close(fig)
    print(str(output.resolve()))


if __name__ == "__main__":
    main()
