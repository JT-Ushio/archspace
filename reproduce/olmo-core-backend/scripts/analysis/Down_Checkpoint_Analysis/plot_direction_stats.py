"""CPU plots for scale-controlled direction statistics; no new model evaluation."""
import argparse
import csv
import json
from pathlib import Path

from plot_token_heatmaps import plt, save, RUNS
import numpy as np

COLORS = dict(S="#d55e00", I="#0072b2", R="#009e73", SR="#a45db0")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.input / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("Analysis incomplete")
    with (args.input / "activations.csv").open() as f:
        rows = list(csv.DictReader(f))
    data = {(r["run"], int(r["layer"]), r["point"]): r for r in rows}
    layers = sorted({int(r["layer"]) for r in rows})
    def values(run, point, metric):
        x = np.array([float(data[run, l, point][metric]) for l in layers])
        if not np.isfinite(x).all():
            raise ValueError("Nonfinite diagnostic")
        return x
    args.output.mkdir(parents=True, exist_ok=False)
    summary = []
    for run in RUNS:
        for point in ("a", "z"):
            summary.append(dict(run=run, point=point, **{
                metric: float(values(run, point, metric).mean()) for metric in (
                    "effective_rank", "direction_effective_rank", "direction_mean_pair_cosine",
                    "near_zero_fraction_mean", "negative_energy_fraction_mean")}))
    with (args.output / "direction_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)
    for name, metrics, titles in (
        ("direction_rank", ("effective_rank", "direction_effective_rank"),
         ("原始幅值：中心化协方差有效秩", "单位长度 token：中心化协方差有效秩")),
        ("direction_similarity", ("direction_mean_pair_cosine", "near_zero_fraction_mean"),
         ("所有非零 token 两两平均余弦相似度", "近零坐标比例：|x| ≤ 0.01 × token RMS"))):
        fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
        for row, point in enumerate(("a", "z")):
            for col, (metric, title) in enumerate(zip(metrics, titles)):
                ax = axes[row, col]
                for run in RUNS:
                    ax.plot(layers, values(run, point, metric), label=run, color=COLORS[run], marker=".")
                ax.set_title(f"{point} · {title}")
                ax.set_xlabel("Transformer 层")
                ax.grid(alpha=.2)
                ax.legend(ncol=4)
        fig.suptitle("Down 方向统计｜区分幅值效应与方向分布\n"
                     "BF16 自然前向；跨样本汇总；零向量排除于方向统计；描述性证据，不是因果证明", fontsize=15)
        save(fig, args.output, name)
    (args.output / "provenance.json").write_text(json.dumps(dict(
        source=str(args.input.resolve()), checkpoint_steps={k: v["step"] for k, v in manifest["checkpoints"].items()},
        sampled_tokens_per_layer=manifest["sampled_tokens_per_layer"],
        averaging="Unweighted mean of per-layer statistics; pair cosine includes tokens from different sequences",
        definition="Participation rank of centered covariance after per-token L2 normalization; zero vectors excluded"), indent=2) + "\n")


if __name__ == "__main__":
    main()
