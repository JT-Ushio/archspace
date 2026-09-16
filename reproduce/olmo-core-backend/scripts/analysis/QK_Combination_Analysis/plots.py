"""Portable SVG charts adapted from the existing checkpoint analysis."""
from html import escape
import math
COLORS = {"L": "#555555", "Q": "#0072B2", "K": "#D55E00", "V": "#009E73", "O": "#CC79A7", "QK": "#E69F00"}

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
        color = COLORS.get(label, list(COLORS.values())[i % len(COLORS)])
        points = " ".join(f"{xp(x):.2f},{yp(y):.2f}" for x, y in sorted(pairs) if math.isfinite(y))
        svg += [f'<polyline points="{points}" stroke="{color}" stroke-width="2" fill="none"/>',
                f'<text x="770" y="{100+i*25}" fill="{color}">{escape(label)}</text>']
    svg += [f'<text x="410" y="502" text-anchor="middle">{escape(xlabel)}</text>',
            f'<text transform="translate(18,265) rotate(-90)" text-anchor="middle">{escape(ylabel)}</text>',
            '</g></svg>']
    path.write_text("\n".join(svg))
