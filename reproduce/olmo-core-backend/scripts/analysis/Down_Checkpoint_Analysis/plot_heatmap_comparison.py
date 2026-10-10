"""Clear paired a/z/delta comparison from measured snapshots, not edited images."""
import argparse
import json
from pathlib import Path

import numpy as np
from plot_token_heatmaps import load_snapshots, plt, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence", type=int, default=0)
    parser.add_argument("--layer", type=int, default=8)
    args = parser.parse_args()
    data, records = load_snapshots(args.input, args.sequence)
    key = f"L{args.layer:02d}"
    nr, nc = data["S"][key + "_a_values"].shape
    ri = np.linspace(0, nr - 1, min(16, nr), dtype=int)
    ci = np.linspace(0, nc - 1, min(64, nc), dtype=int)
    positions = data["S"]["positions"][ri]
    channels = data["S"][key + "_a_channels"][ci]
    fig, axes = plt.subplots(2, 3, figsize=(16, 7.8))
    fig.subplots_adjust(left=.085, right=.95, top=.79, bottom=.19, wspace=.16, hspace=.62)
    info = []
    for row, (run, name) in enumerate((("S", "SiLU"), ("R", "RMSNorm"))):
        arrays = []
        for point in ("a", "z"):
            x = data[run][key + f"_{point}_values"].astype(np.float64)
            rms = data[run][key + f"_{point}_metrics"][:, :1]
            arrays.append(np.divide(x, rms, out=np.zeros_like(x), where=rms > 0))
        a, z = arrays
        delta = z - a
        valid = (np.linalg.norm(a, axis=1) > 0) & (np.linalg.norm(z, axis=1) > 0)
        cosine = np.sum(a[valid] * z[valid], axis=1) / (np.linalg.norm(a[valid], axis=1) * np.linalg.norm(z[valid], axis=1))
        mean_change = float(np.abs(delta).mean())
        text = f"{name}    平均 |Δ| = {mean_change:.6f}；a/z 平均余弦 = {cosine.mean():.6f}"
        fig.text(.085, .835 if row == 0 else .455, text, fontsize=13, weight="bold", color="#263648")
        for col, (x, title) in enumerate(zip((a, z, delta), ("激活前 a", "激活后 z", "变化 Δ = 激活后 − 激活前"))):
            limit = 1.5 if col == 2 else 3
            shown = x[np.ix_(ri, ci)]
            ax = axes[row, col]
            im = ax.imshow(shown, cmap="RdBu_r", vmin=-limit, vmax=limit,
                           interpolation="nearest", aspect="auto")
            ax.set_title(title, fontsize=12, pad=8)
            xt = np.array([0, len(ci)//2, len(ci)-1])
            yt = np.array([0, len(ri)//2, len(ri)-1])
            ax.set_xticks(xt, labels=channels[xt], fontsize=9)
            ax.set_yticks(yt, labels=positions[yt], fontsize=9)
            ax.set_xlabel("通道编号", fontsize=10)
            if col == 0:
                ax.set_ylabel("token 原位置", fontsize=10)
            if row == 1 and col in (1, 2):
                barax = fig.add_axes([.20 if col == 1 else .665, .115, .30 if col == 1 else .24, .017])
                bar = fig.colorbar(im, cax=barax, orientation="horizontal", ticks=[-limit, 0, limit])
                bar.set_label("前 / 后共用色标" if col == 1 else "变化列共用色标", fontsize=10)
            info.append(dict(run=run, panel=title, clipped_fraction=float((np.abs(shown) > limit).mean()),
                             color_limit=limit, mean_abs_delta_all_saved_tokens=mean_change,
                             mean_cosine_all_saved_tokens=float(cosine.mean())))
    fig.suptitle(f"SiLU 与 RMSNorm：谁改变了 token 的特征结构？", fontsize=20, weight="bold", y=.985)
    fig.text(.5, .918, f"第 {args.layer} 层 · 同一批 token · 左到右看变化，重点比较最后一列", ha="center", fontsize=13)
    fig.text(.085, .015,
             "每个 token 的 a、z 分别除以全 512 维 RMS；白色≈0，红色为正，蓝色为负。\n"
             "等距展示 16 token × 64 通道；标题统计用全部已保存 token/512 通道。色标外数值仅显示截断；跨模型通道语义不对齐。",
             fontsize=9, color="#485463")
    args.output.mkdir(parents=True, exist_ok=False)
    save(fig, args.output, "silu_vs_rms_comparison")
    (args.output / "provenance.json").write_text(json.dumps(dict(
        source=str(args.input.resolve()), sample=records["S"], layer=args.layer,
        token_positions=positions.tolist(), channels=channels.tolist(), panels=info,
        note="BF16 natural forward. Original arrays unchanged. Delta compares separately RMS-normalized features, not raw amplitudes."),
        indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
