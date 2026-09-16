"""Validate complete paired coverage before releasing initialization artifacts."""
import csv
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

MODES = ("independent", "svd_raw", "svd_scale_matched")
PRECISIONS = ("bf16", "fp8")


def report(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    samples = json.loads((root / "samples.json").read_text())["samples"]
    ids = {str(s["sample"]) for s in samples}
    pairs = set(manifest["pairs"])
    ce, acts, weights, seen = [], [], [], set()
    for rank in range(manifest["world_size"]):
        done = json.loads((root / f"rank-{rank}.complete.json").read_text())
        if done.get("fsdp_initialization_smoke") != "passed":
            raise ValueError("FSDP initialization smoke has not passed on every worker")
        current = {str(s) for s in done["samples"]}
        if current & seen:
            raise ValueError("Duplicate samples across workers")
        seen |= current
        for stem, target in (("ce", ce), ("activations", acts), ("weights", weights)):
            with (root / f"{stem}-rank{rank}.csv").open() as stream:
                target.extend(csv.DictReader(stream))
    if seen != ids:
        raise ValueError("Incomplete sample coverage")
    for rows, keys, expected in (
        (ce, ("mode", "precision", "sample"), set(itertools.product(MODES, PRECISIONS, ids))),
        (acts, ("mode", "precision", "pair", "stage", "sample"),
         set(itertools.product(MODES, PRECISIONS, pairs, ("a", "z", "y"), ids))),
        (weights, ("mode", "pair"), set(itertools.product(MODES, pairs))),
    ):
        actual = [tuple(row[key] for key in keys) for row in rows]
        if len(actual) != len(set(actual)) or set(actual) != expected:
            raise ValueError(f"Missing/duplicate measurements: {keys}")
    for rows, names in ((ce, ("ce", "tokens")), (acts, ("rms", "abs_max", "mean")),
                        (weights, ("E_rms", "D_rms", "DE_frobenius"))):
        if any(not math.isfinite(float(row[name])) for row in rows for name in names):
            raise ValueError("Nonfinite measurements")
    for name, rows in (("ce", ce), ("activations", acts), ("weights", weights)):
        with (root / f"{name}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    groups = defaultdict(list)
    for row in ce:
        groups[row["mode"], row["precision"]].append(row)
    lines = ["# SVD initialization: step0 checks", "",
             f"Complete: {len(pairs)} projection pairs; {len(ids)} fixed sequences; {len(acts)} activation records.",
             "", "| Initialization | Precision | Sampled-token CE |", "|---|---|---:|"]
    for (mode, precision), rows in sorted(groups.items()):
        loss = sum(float(r["ce"])*int(r["tokens"]) for r in rows)/sum(int(r["tokens"]) for r in rows)
        lines.append(f"| {mode} | {precision} | {loss:.6f} |")
    lines.extend(["", "These are step0 sampled-token diagnostics, not full validation or training results.",
                  "FP8 and BF16 use the same weights and samples; all three initializations retain the nonlinear architecture.",
                  "Scale matching matches each factor's weight RMS, not activation distributions or the unscaled dense matrix.",
                  "Raw CSVs retain per-sequence/per-layer values; spectra-rank*.pt contains factor/product singular values."])
    (root / "REPORT.md").write_text("\n".join(lines)+"\n")
    summary = defaultdict(list)
    for row in acts:
        summary[row["mode"], row["precision"], row["pair"], row["stage"]].append(row)
    with (root / "activation_summary.csv").open("w", newline="") as stream:
        names = ("rms", "negative_fraction", "abs_p99", "abs_p999", "abs_max")
        writer = csv.writer(stream)
        writer.writerow(("mode", "precision", "pair", "stage", *[f"mean_{n}" for n in names]))
        for key, rows in sorted(summary.items()):
            writer.writerow((*key, *[mean(float(r[n]) for r in rows) for n in names]))
    # Repository-native SVG plots; no image generation or extra plotting dependency.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "QK_Combination_Analysis"))
    from plots import chart
    labels = {"independent": "A independent", "svd_raw": "B raw SVD", "svd_scale_matched": "C matched SVD"}
    for precision, projection, stage in itertools.product(PRECISIONS, ("q", "k", "v", "o", "gate", "up", "down"), ("a", "z", "y")):
        series = defaultdict(list)
        for (mode, prec, pair, stg), rows in summary.items():
            if prec == precision and pair.split(".")[-1] == projection and stg == stage:
                series[labels[mode]].append((int(pair.split(".")[1]), mean(float(r["rms"]) for r in rows)))
        chart(root / f"{precision}-{projection}-{stage}-rms.svg", f"{precision}: {projection} {stage} RMS at step0",
              dict(sorted(series.items())), "Layer index", "Mean per-sequence RMS")
