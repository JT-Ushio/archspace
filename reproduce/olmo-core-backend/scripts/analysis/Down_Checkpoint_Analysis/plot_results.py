"""Produce portable SVG plots and combined CSVs without matplotlib installation."""
import argparse
import csv
from html import escape
import json
import math
from pathlib import Path

import torch

COLORS = {"S": "#d95f02", "I": "#1b9e77", "R": "#7570b3", "SR": "#e7298a"}


def read_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def chart(path, title, series, xlabel, ylabel):
    """series: label -> [(x,y),...]. Axes are explicit; no smoothing."""
    values = [pair for pairs in series.values() for pair in pairs if all(math.isfinite(v) for v in pair)]
    if not values:
        return
    xmin, xmax = min(x for x, _ in values), max(x for x, _ in values)
    ymin, ymax = min(y for _, y in values), max(y for _, y in values)
    if xmin == xmax:
        xmax += 1
    span = ymax - ymin
    ymin -= .05 * (span or 1)
    ymax += .05 * (span or 1)
    xp = lambda x: 85 + (x - xmin) / (xmax - xmin) * 655
    yp = lambda y: 440 - (y - ymin) / (ymax - ymin) * 360
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="960" height="520" viewBox="0 0 960 520">',
           '<rect width="960" height="520" fill="white"/>',
           '<g font-family="sans-serif" font-size="13" fill="#222">',
           f'<text x="85" y="32" font-size="18">{escape(title)}</text>']
    for i in range(6):
        y = ymin + (ymax - ymin) * i / 5
        svg += [f'<line x1="85" y1="{yp(y)}" x2="740" y2="{yp(y)}" stroke="#eee"/>',
                f'<text x="77" y="{yp(y)+4}" text-anchor="end">{y:.3g}</text>']
        x = xmin + (xmax - xmin) * i / 5
        svg.append(f'<text x="{xp(x)}" y="462" text-anchor="middle">{x:.3g}</text>')
    for i, (label, pairs) in enumerate(series.items()):
        color = COLORS.get(label, list(COLORS.values())[i % 4])
        points = " ".join(f"{xp(x):.2f},{yp(y):.2f}" for x, y in sorted(pairs) if math.isfinite(y))
        svg += [f'<polyline points="{points}" stroke="{color}" stroke-width="2" fill="none"/>',
                f'<text x="770" y="{100+i*25}" fill="{color}">{escape(label)}</text>']
    svg += [f'<text x="410" y="502" text-anchor="middle">{escape(xlabel)}</text>',
            f'<text transform="translate(18,265) rotate(-90)" text-anchor="middle">{escape(ylabel)}</text>',
            '</g></svg>']
    path.write_text("\n".join(svg))


def build_plots(directory):
    directory = Path(directory)
    plots = directory / "plots"
    plots.mkdir(exist_ok=True)
    combined = {}
    for name in ("activations", "replay", "spectra", "silu_sensitivity", "per_sequence"):
        rows = [r for label in COLORS for r in read_csv(directory / label / f"{name}.csv")]
        combined[name] = rows
        fields = list(dict.fromkeys(k for r in rows for k in r))
        with (directory / f"{name}.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    rows = combined["activations"]
    for point in ("h", "a", "z", "y", "o"):
        for metric in ("rms_p50", "rms_p99", "absmax_over_rms_p99", "negative_fraction_mean",
                       "negative_energy_fraction_mean", "top1pct_energy_fraction_mean"):
            series = {label: [(int(r["layer"]), float(r[metric])) for r in rows if r["run"] == label and r["point"] == point]
                      for label in COLORS}
            chart(plots / f"{point}_{metric}.svg", f"Natural BF16: {point} / {metric}", series, "Layer (1-based)", metric)
    chart(plots / "silu_small_derivative.svg", "Hypothetical SiLU at recorded a: |derivative| < 0.1",
          {label: [(int(r["layer"]), float(r["mean"])) for r in combined["silu_sensitivity"] if r["run"] == label]
           for label in COLORS}, "Layer (1-based)", "Mean fraction of coordinates")
    for point in ("a", "z"):
        chart(plots / f"{point}_effective_rank.svg", f"Centered covariance participation rank: {point}",
              {label: [(int(r["layer"]), float(r["effective_rank"])) for r in rows if r["run"] == label and r["point"] == point]
               for label in COLORS}, "Layer (1-based)", "Effective rank")
    for left, right in (("I", "R"), ("I", "S"), ("S", "SR")):
        for point in ("z", "y", "o"):
            for metric in ("one_minus_cosine_mean", "relative_l2_mean"):
                chart(plots / f"replay_{left}_{right}_{point}_{metric}.svg",
                      f"FP32 local replay: {left} vs {right}, {point}",
                      {label: [(int(r["layer"]), float(r[metric])) for r in combined["replay"]
                               if r["run"] == label and r["left"] == left and r["right"] == right and r["point"] == point and r.get(metric)]
                       for label in COLORS}, "Layer (1-based); legend = source checkpoint", metric)
    samples = {label: torch.load(directory / label / "histogram_samples.pt", weights_only=True, map_location="cpu") for label in COLORS}
    layers = sorted({int(r["layer"]) for r in rows})
    for layer in layers:
        for point in ("a", "z"):
            transformed = {label: torch.sign(v[f"{layer}_{point}"]) * torch.log1p(v[f"{layer}_{point}"].abs()) for label, v in samples.items()}
            lo = min(v.min().item() for v in transformed.values())
            hi = max(v.max().item() for v in transformed.values())
            if hi == lo:
                hi += 1
            centers = torch.linspace(lo, hi, 65)
            centers = (centers[:-1] + centers[1:]) / 2
            series = {label: list(zip(centers.tolist(), (torch.histc(v, bins=64, min=lo, max=hi) / len(v)).tolist())) for label, v in transformed.items()}
            chart(plots / f"layer{layer:02d}_{point}_histogram.svg", f"Layer {layer} / {point}: sampled raw activation distribution",
                  series, "sign(x) * log(1 + |x|); shared bins across runs", "Probability per transformed bin")
            chart(plots / f"layer{layer:02d}_{point}_spectrum.svg", f"Layer {layer} / {point}: centered covariance spectrum",
                  {label: [(int(r["component"]), math.log10(max(float(r["fraction"]), 1e-12))) for r in combined["spectra"]
                           if r["run"] == label and int(r["layer"]) == layer and r["point"] == point] for label in COLORS},
                  "Component", "log10(explained variance fraction), floor -12")
    manifest = json.loads((directory / "manifest.json").read_text())
    links = "\n".join(f"- [{p.stem}](plots/{p.name})" for p in sorted(plots.glob("*.svg")))
    (directory / "report.md").write_text(
        "# Down checkpoint 起步分析\n\n"
        f"实际有效输入 tokens：{manifest['valid_tokens']:,}；每层统计采样 tokens：{manifest['sampled_tokens_per_layer']:,}。\n\n"
        "自然前向：BF16；局部重放：从相同的已记录 a 出发，FP32、TF32 关闭。"
        "没有进行 FP8 kernel 误差测量、训练或验证 loss 重评。\n\n"
        "h=Down 输入，a=encoder 输出，z=中间激活输出，y=decoder 输出，o=外部 RMSNorm 输出。"
        "padding 不计入统计；每个样本随机选择固定位置，四个模型位置完全一致。"
        "验证集按源文件平衡抽样，短文档 padding 后仍可能远少于 8192 个有效 tokens。\n\n"
        "## 先看什么\n\n"
        "1. replay I/R 与 S/SR 的 o 误差，检验外部归一化是否消除了大部分尺度差异。\n"
        "2. replay I/S 的 z/y/o 方向差异及其层级分布。\n"
        "3. a/z 的负值能量、SiLU 小导数比例、有效秩与谱，寻找可进一步验证的线索。\n\n"
        "自然统计来自不同训练权重，不能直接作为激活的因果效应；局部重放固定输入权重，"
        "但也不等于替换激活后重新训练的表现。跨模型不做逐通道语义对齐。"
        "RMSNorm 后尺度接近 1 是定义，不是效果更好的证据；不预设秩或重尾变化的方向。\n\n"
        "## 数据\n\n"
        "- activations.csv：采样 token 的统计（均值/P10/P50/P90/P99）；sample_abs_* 来自更小的坐标样本。\n"
        "- replay.csv：同输入同 decoder 下四种激活两两比较；relative_l2 分母为 left，零向量单独计数。\n"
        "- spectra.csv：a/z 的中心化协方差谱；非负截断前负特征值质量另记在 activations.csv。\n"
        "- silu_sensitivity.csv：各模型 a 上假想 SiLU 导数小于阈值的比例，不是实测训练梯度。\n"
        "- per_sequence.csv：逐样本统计，sample_order 对应 samples.json 顺序，可用于后续配对分析。\n"
        "- samples.json / manifest.json：样本位置与哈希、checkpoint、配置与代码哈希、环境及完成状态。\n\n"
        "## 图表\n\n" + links + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    build_plots(parser.parse_args().directory)
