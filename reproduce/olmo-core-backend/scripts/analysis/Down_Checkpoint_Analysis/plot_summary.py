"""CPU-only summary figures from existing checkpoint-analysis CSV files."""
import argparse
import csv
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mose-mpl-"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

RUNS = ("S", "I", "R", "SR")
COLORS = dict(S="#d55e00", I="#0072b2", R="#009e73", SR="#a45db0")
LABELS = dict(S="S · SiLU", I="I · Identity", R="R · RMSNorm", SR="SR · SiLU → RMSNorm")


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.input / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("Analysis must be complete before plotting")
    acts = read_csv(args.input / "activations.csv")
    replay = read_csv(args.input / "replay.csv")
    deriv = read_csv(args.input / "silu_sensitivity.csv")
    a = {(r["run"], int(r["layer"]), r["point"]): r for r in acts}
    rp = {(r["run"], int(r["layer"]), r["left"], r["right"], r["point"]): r for r in replay}
    ds = {(r["run"], int(r["layer"])): r for r in deriv}
    layers = sorted({int(r["layer"]) for r in acts})
    assert len(a) == len(RUNS) * len(layers) * 5
    for run in RUNS:
        for layer in layers:
            for point in ("h", "a", "z", "y", "o"):
                assert int(a[run, layer, point]["sampled_tokens"]) == manifest["sampled_tokens_per_layer"]
    font = Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf")
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = ["DejaVu Sans", font_manager.FontProperties(fname=str(font)).get_name()]
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.unicode_minus": False, "figure.facecolor": "#fafbfc",
                         "axes.facecolor": "white", "svg.fonttype": "none"})
    args.output.mkdir(parents=True, exist_ok=False)
    x = np.arange(len(RUNS))
    def values(run, point, metric):
        v = np.array([float(a[run, l, point][metric]) for l in layers])
        assert np.isfinite(v).all()
        return v
    def distances(run, pair, metric):
        return np.array([float(rp[(run, l, *pair, "o")][metric]) for l in layers])
    def style(ax, title, ylabel):
        ax.set_title(title, loc="left", pad=15)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
    def save(fig, name, title, note):
        fig.suptitle(title, fontsize=20, weight="bold", x=.07, ha="left", y=.98)
        fig.text(.07, .025, note, fontsize=10, color="#505b66", va="bottom")
        fig.tight_layout(rect=(.02, .095, .99, .925), h_pad=3, w_pad=3)
        for suffix in ("png", "svg", "pdf"):
            fig.savefig(args.output / f"{name}.{suffix}", dpi=160)
        plt.close(fig)
    def profile(ax, point, metric, title, ylabel, scale=1, log=False):
        for run in RUNS:
            ax.plot(layers, values(run, point, metric)*scale, color=COLORS[run], label=LABELS[run], marker="o", ms=4, lw=2)
        style(ax, title, ylabel)
        ax.set_xlabel("Transformer 层")
        ax.set_xticks([1, 4, 8, 12, 16])
        if log:
            ax.set_yscale("log")
        ax.legend(fontsize=9, frameon=False)

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    ax = axs[0, 0]
    pairs = [("I", "R"), ("S", "SR"), ("I", "S")]
    for j, pair in enumerate(pairs):
        v = [distances(run, pair, "relative_l2_mean").mean() for run in RUNS]
        bars = ax.bar(x + (j-1)*.24, v, width=.23, label=" ↔ ".join(pair), color=["#0072b2", "#009e73", "#d55e00"][j])
        ax.bar_label(bars, labels=[f"{n:.1e}" if n<.001 else f"{n:.2f}" for n in v], padding=14 if j==0 else 3, fontsize=8)
    ax.set_yscale("log"); ax.set_ylim(3e-8, 100)
    ax.set_xticks(x, RUNS); ax.set_xlabel("提供相同 a、B 的 checkpoint")
    style(ax, "① 外部 Norm 后：归一化差异近零，SiLU 差异仍大", "平均相对 L2 误差（对数轴）")
    ax.legend(frameon=False, ncol=3, fontsize=9, loc="upper left")
    ax = axs[0, 1]
    for j, point in enumerate(("a", "z")):
        v = [values(run, point, "negative_energy_fraction_mean").mean()*100 for run in RUNS]
        bars = ax.bar(x+(j-.5)*.32, v, .31, color=["#8c9cab", "#354b61"][j], label=["激活前 a", "激活后 z"][j])
        ax.bar_label(bars, labels=["≈0" if n<1e-8 else f"{n:.1f}%" for n in v], padding=4, fontsize=10)
    ax.set_xticks(x, RUNS); ax.set_ylim(0, 65)
    style(ax, "② SiLU 几乎消除负坐标的输出能量", "负值能量占比（%）")
    ax.legend(frameon=False, ncol=2)
    ax = axs[1, 0]
    for j, point in enumerate(("a", "z")):
        v = [values(run, point, "effective_rank").mean() for run in RUNS]
        bars = ax.bar(x+(j-.5)*.32, v, .31, color=["#8c9cab", "#354b61"][j], label=["激活前 a", "激活后 z"][j])
        ax.bar_label(bars, fmt="%.1f", padding=4, fontsize=10)
    ax.set_xticks(x, RUNS); ax.set_ylim(0, 120)
    style(ax, "③ 协方差有效秩：不能直接等同于模型能力", "Participation ratio（不是矩阵代数秩）")
    ax.legend(frameon=False, ncol=2)
    ax = axs[1, 1]
    for run in RUNS:
        ax.plot(layers, distances(run, ("I", "S"), "relative_l2_mean"), label=LABELS[run], color=COLORS[run], marker="o", ms=4, lw=2)
    style(ax, "④ 替换 SiLU 的局部影响遍及各层", "I ↔ S：外部 Norm 后平均相对 L2")
    ax.set_xlabel("Transformer 层"); ax.set_xticks([1,4,8,12,16]); ax.set_ylim(0,.85); ax.legend(frameon=False, fontsize=9)
    save(fig, "01_summary", "Down 激活诊断｜先看这四个对比",
         "S=SiLU；I=Identity；R=RMSNorm；SR=RMSNorm(SiLU)。柱状图为 16 层等权平均，不是跨 seed 统计。\n"
         "自然前向 BF16；同输入同权重的局部重放 FP32。不是 FP8 误差测量，也不是替换后的全模型 loss。")

    fig, axs = plt.subplots(2, 2, figsize=(14,10))
    profile(axs[0,0], "a", "negative_fraction_mean", "① SiLU 组深层 a 更偏向正值", "负坐标比例（%）", 100)
    profile(axs[0,1], "z", "negative_energy_fraction_mean", "② Norm 无法恢复被 SiLU 抑制的负值", "激活后负值能量占比（%）", 100)
    profile(axs[1,0], "a", "effective_rank", "③ 激活前：各层谱差异明显", "协方差有效秩")
    profile(axs[1,1], "z", "effective_rank", "④ 激活后：归一化也会改变协方差谱", "协方差有效秩")
    save(fig, "02_layer_structure", "逐层结构｜符号分布与协方差谱",
         f"所有模型采用相同样本及位置；每层采样 {manifest['sampled_tokens_per_layer']:,} 个有效 token。曲线来自独立训练的不同权重。\n"
         "这些是描述性关联；有效秩受样本幅度影响，不能据此认定信息恢复、容量提升或因果关系。")

    fig, axs = plt.subplots(2,2,figsize=(14,10))
    profile(axs[0,0], "a", "rms_p50", "① encoder 输出 a：幅度很大", "token RMS 中位数（对数轴）", log=True)
    profile(axs[0,1], "y", "rms_p50", "② decoder 输出 y：内部 Norm 改变尺度", "token RMS 中位数（对数轴）", log=True)
    profile(axs[1,0], "h", "top1pct_energy_fraction_mean", "③ Down 输入能量集中在少量通道", "每 token 最大 1% 通道能量占比（%）", 100)
    ax=axs[1,1]
    for run in RUNS:
        ax.plot(layers,[100*float(ds[run,l]["mean"]) for l in layers],color=COLORS[run],label=LABELS[run],marker="o",ms=4,lw=2)
    style(ax,"④ 在记录的 a 上假想施加 SiLU", "|SiLU 导数| < 0.1 的坐标比例（%）")
    ax.set_xlabel("Transformer 层"); ax.set_xticks([1,4,8,12,16]); ax.legend(frameon=False,fontsize=9)
    save(fig,"03_scale_and_saturation","尺度与饱和｜为什么此处 SiLU 接近单边门控",
         "这是 FP8 训练 checkpoint 的 BF16 分析前向，不能直接代表 FP8 训练中的数值误差。\n"
         "导数比例是局部函数诊断，不是实测梯度消失率；归一化后的尺度变小也不等于性能变好。")
    provenance = {"input": str(args.input.resolve()), "checkpoint_steps": {k:v["step"] for k,v in manifest["checkpoints"].items()},
                  "sampled_tokens_per_layer": manifest["sampled_tokens_per_layer"], "aggregation":"layer-equal mean; no seed error bars"}
    (args.output/"provenance.json").write_text(json.dumps(provenance,ensure_ascii=False,indent=2))
    print(args.output.resolve())


if __name__ == "__main__":
    main()
