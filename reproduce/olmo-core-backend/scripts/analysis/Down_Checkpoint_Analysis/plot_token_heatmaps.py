"""Plot saved token snapshots on CPU. No checkpoint loading or GPU required."""
import argparse
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mose-heatmap-mpl-"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

RUNS = ("S", "I", "R", "SR")
POINTS = ("h", "a", "z", "y", "o")
FONT = Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf")
if FONT.exists():
    font_manager.fontManager.addfont(str(FONT))
    plt.rcParams["font.family"] = ["DejaVu Sans", font_manager.FontProperties(fname=str(FONT)).get_name()]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["svg.fonttype"] = "none"


def load_snapshots(root, order):
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("Analysis manifest must be complete")
    data, records = {}, {}
    for run in RUNS:
        stem = root / run / "token_features" / f"sequence_{order:03d}"
        with np.load(stem.with_suffix(".npz"), allow_pickle=False) as archive:
            data[run] = {k: archive[k] for k in archive.files}
        records[run] = json.loads(stem.with_suffix(".json").read_text())
        if run != "S":
            for key in ("positions", "token_ids"):
                if not np.array_equal(data[run][key], data["S"][key]):
                    raise ValueError(f"Mismatched token alignment: {run}/{key}")
            for key in ("input_sha256", "dataset_index", "source"):
                if records[run][key] != records["S"][key]:
                    raise ValueError(f"Mismatched sample: {run}/{key}")
            if set(data[run]) != set(data["S"]):
                raise ValueError("Mismatched feature keys")
        for key, value in data[run].items():
            if not np.isfinite(value).all():
                raise ValueError(f"Nonfinite snapshot {run}/{key}")
            if run != "S" and (value.shape != data["S"][key].shape or
                               (key.endswith("_channels") and not np.array_equal(value, data["S"][key]))):
                raise ValueError(f"Mismatched feature shape/channels: {run}/{key}")
    return data, records


def save(fig, output, name):
    for ext in ("png", "svg", "pdf"):
        fig.savefig(output / f"{name}.{ext}", dpi=160, bbox_inches="tight")
    plt.close(fig)


def token_axis(ax, positions):
    ticks = np.unique(np.linspace(0, len(positions) - 1, min(6, len(positions)), dtype=int))
    ax.set_yticks(ticks, labels=positions[ticks])
    ax.set_ylabel("原序列 token 位置")


def build(root, output, order=0, selected_layers=(1, 4, 8, 12, 16)):
    data, records = load_snapshots(root, order)
    layers = sorted(int(k.split("_")[0][1:]) for k in data["S"] if k.endswith("_a_values"))
    if not set(selected_layers).issubset(layers):
        raise ValueError(f"Requested layers not available; available={layers}")
    output.mkdir(parents=True, exist_ok=False)
    positions = data["S"]["positions"]
    # Exact original positions on y axis; no interpolation across sampled tokens.
    for metric, title, cmap in ((0, "特征强度 log10(RMS)", "magma"),
                                (1, "负值坐标比例", "viridis"),
                                (2, "负值能量占比", "viridis")):
        fig, axes = plt.subplots(4, 5, figsize=(19, 15), layout="constrained")
        for col, point in enumerate(POINTS):
            matrices = []
            for run in RUNS:
                a = np.stack([data[run][f"L{l:02d}_{point}_metrics"][:, metric] for l in layers], axis=1)
                matrices.append(np.log10(np.maximum(a, 1e-12)) if metric == 0 else a)
            lo, hi = (min(a.min() for a in matrices), max(a.max() for a in matrices)) if metric == 0 else (0, 1)
            if lo == hi:
                hi = lo + 1e-6
            for row, (run, matrix) in enumerate(zip(RUNS, matrices)):
                ax = axes[row, col]
                im = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap=cmap, vmin=lo, vmax=hi)
                ax.set_title(f"{run} · {point}")
                ticks = np.unique(np.linspace(0, len(layers) - 1, min(5, len(layers)), dtype=int))
                ax.set_xticks(ticks, labels=np.asarray(layers)[ticks])
                ax.set_xlabel("Transformer 层")
                token_axis(ax, positions)
            fig.colorbar(im, ax=axes[:, col], shrink=.65)
        fig.suptitle(f"同一批 token 的逐层变化｜{title}\nS=SiLU · I=Identity · R=RMSNorm · SR=SiLU→RMSNorm\n"
                     "每列跨模型共用色标；统计使用全部通道；BF16 自然前向", fontsize=16)
        save(fig, output, f"sequence_{order:03d}_token_layer_{metric}")

    # a/z coordinate semantics align within a model/layer, not across trained models.
    # Two complementary views prevent raw scales from hiding signs and sparsity.
    raw_max = max(np.log10(1 + np.abs(data[r][f"L{l:02d}_{p}_values"].astype(np.float64))).max()
                  for r in RUNS for l in selected_layers for p in ("a", "z"))
    clipping = []
    for run in RUNS:
        for view in ("rms_normalized", "signed_log"):
            fig, axes = plt.subplots(len(selected_layers), 2, figsize=(16, 3.5 * len(selected_layers)),
                                     squeeze=False, layout="constrained")
            vmax = 4 if view == "rms_normalized" else max(raw_max, 1e-6)
            for row, layer in enumerate(selected_layers):
                for col, point in enumerate(("a", "z")):
                    key = f"L{layer:02d}_{point}"
                    x = data[run][key + "_values"].astype(np.float64)
                    rms = data[run][key + "_metrics"][:, :1]
                    if view == "rms_normalized":
                        x = np.divide(x, rms, out=np.zeros_like(x), where=rms > 0)
                    else:
                        x = np.sign(x) * np.log10(1 + np.abs(x))
                    clipped = float((np.abs(x) > vmax).mean())
                    clipping.append(dict(run=run, layer=layer, point=point, view=view, clipped_fraction=clipped))
                    ax = axes[row, col]
                    im = ax.imshow(x, aspect="auto", interpolation="nearest", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
                    ax.set_title(f"Layer {layer} · {point} · 色标截断 {clipped:.1%}")
                    channels = data[run][key + "_channels"]
                    ticks = np.unique(np.linspace(0, len(channels) - 1, min(6, len(channels)), dtype=int))
                    ax.set_xticks(ticks, labels=channels[ticks])
                    ax.set_xlabel("原始通道编号（不重排）")
                    token_axis(ax, positions)
            description = "逐 token 除以全通道 RMS，统一色标 ±4" if view == "rms_normalized" else "sign(x) × log10(1+|x|)，跨模型/层共用色标"
            fig.suptitle(f"{run}｜Down 激活前 a → 激活后 z\n{description}\n"
                         "零值为白色；同一行对应同一 token；跨模型通道语义不对齐", fontsize=16)
            fig.colorbar(im, ax=axes.ravel().tolist(), shrink=.6)
            save(fig, output, f"sequence_{order:03d}_{run}_{view}")
    (output / "provenance.json").write_text(json.dumps(dict(
        source=str(root.resolve()), sample=records["S"], layers=selected_layers,
        clipping=clipping, note="Sampled positions, not contiguous tokens; no tokenizer/text decoding; descriptive BF16 activations, not FP8 error."),
        indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence", type=int, default=0, help="Zero-based sample order")
    parser.add_argument("--layers", type=int, nargs="+", default=[1, 4, 8, 12, 16])
    args = parser.parse_args()
    if not args.layers or len(set(args.layers)) != len(args.layers) or args.sequence < 0:
        parser.error("Require unique layers and nonnegative sequence order")
    build(args.input, args.output, args.sequence, args.layers)
