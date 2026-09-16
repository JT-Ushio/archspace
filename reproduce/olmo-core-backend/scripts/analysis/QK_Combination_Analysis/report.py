"""Aggregate verified disjoint worker outputs; paired sample CE and SVG figures."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import random
import statistics

from plots import chart


def read_rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def write_rows(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def paired_ci(values, seed=34521):
    """Paired sequence bootstrap, conditional on this seed/model/dataset sample."""
    rng = random.Random(seed)
    means = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(2000))
    return means[50], means[1949]


def build_report(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    samples = json.loads((root / "samples.json").read_text())
    if list(root.glob("rank-*.error.json")):
        raise ValueError("An analysis worker failed; refuse partial comparison")
    observed = []
    for rank in range(manifest["world_size"]):
        observed += json.loads((root / f"rank-{rank:02d}.complete.json").read_text())["samples"]
    if sorted(observed) != list(range(len(samples))):
        raise ValueError("Missing or duplicate worker sample assignments")
    steps = manifest["arguments"]["steps"]
    token_lookup, groups = {}, defaultdict(list)
    attention_groups = defaultdict(list)
    for rank in range(manifest["world_size"]):
        worker = root / f"rank-{rank:02d}"
        for label in "LQK":
            for step in steps:
                for precision in ("bf16", "fp8"):
                    for row in read_rows(worker / f"{label}-{step}-{precision}-tokens.csv"):
                        sid, position = int(row["sample"]), int(row["position"])
                        key = (label, step, precision, sid, position)
                        if key in token_lookup or position not in samples[sid]["positions"]:
                            raise ValueError("Duplicate/unexpected token position")
                        token_lookup[key] = (int(row["target_id"]), float(row["ce"]))
                        groups[label, step, precision, sid].append(float(row["ce"]))
                    for row in read_rows(worker / f"{label}-{step}-{precision}-attention.csv"):
                        for metric in ("normalized_entropy", "entropy", "mean_distance", "logit_std", "max_probability", "distance_over1024_mass"):
                            key = (label, step, precision, int(row["layer"]), metric)
                            attention_groups[key].append(float(row[metric]))
    sequence_rows, scores = [], {}
    for label in "LQK":
        for step in steps:
            for precision in ("bf16", "fp8"):
                for sample in samples:
                    sid = sample["sample"]
                    for position in sample["positions"]:
                        key = (label, step, precision, sid, position)
                        reference = ("L", step, "bf16", sid, position)
                        if key not in token_lookup or token_lookup[key][0] != token_lookup[reference][0]:
                            raise ValueError("Missing/misaligned scored token")
                    value = statistics.mean(groups[label, step, precision, sid])
                    scores[label, step, precision, sid] = value
                    sequence_rows.append({"variant": label, "step": step, "precision": precision,
                                          "sample": sid, "source": sample["source"], "mean_ce": value,
                                          "scored_tokens": len(sample["positions"])})
    write_rows(root / "per_sequence_ce.csv", sequence_rows)
    summaries, effects = [], []
    for step in steps:
        for precision in ("bf16", "fp8"):
            for label in "LQK":
                values = [scores[label, step, precision, s["sample"]] for s in samples]
                summaries.append({"variant": label, "step": step, "tokens": step * 2097152,
                                  "precision": precision, "sequence_macro_ce": statistics.mean(values)})
            for label in "QK":
                gains = [scores["L", step, precision, s["sample"]] - scores[label, step, precision, s["sample"]] for s in samples]
                low, high = paired_ci(gains)
                effects.append({"comparison": f"L_minus_{label}", "step": step, "precision": precision,
                                "mean_delta_ce": statistics.mean(gains), "ci025": low, "ci975": high,
                                "sequence_win_fraction": sum(x > 0 for x in gains) / len(gains)})
        for label in "LQK":
            gaps = [scores[label, step, "fp8", s["sample"]] - scores[label, step, "bf16", s["sample"]] for s in samples]
            low, high = paired_ci(gaps)
            effects.append({"comparison": f"{label}_fp8_minus_bf16", "step": step, "precision": "paired",
                            "mean_delta_ce": statistics.mean(gaps), "ci025": low, "ci975": high,
                            "sequence_win_fraction": sum(x > 0 for x in gaps) / len(gaps)})
    write_rows(root / "summary.csv", summaries)
    write_rows(root / "paired_effects.csv", effects)
    attention_rows = [{"variant": key[0], "step": key[1], "precision": key[2], "layer": key[3],
                       "metric": key[4], "mean": statistics.mean(values)} for key, values in attention_groups.items()]
    write_rows(root / "attention_summary.csv", attention_rows)
    plots = root / "plots"
    plots.mkdir(exist_ok=True)
    for precision in ("bf16", "fp8"):
        chart(plots / f"ce-{precision}.svg", f"Fixed sample CE ({precision}); lower is better",
              {label: [(r["tokens"] / 1e9, r["sequence_macro_ce"]) for r in summaries if r["variant"] == label and r["precision"] == precision] for label in "LQK"},
              "Training tokens (B)", "Equal-sequence mean CE")
        chart(plots / f"gain-{precision}.svg", f"Gain over linear ({precision}); positive is better",
              {label: [(r["step"] * 2097152 / 1e9, r["mean_delta_ce"]) for r in effects if r["comparison"] == f"L_minus_{label}" and r["precision"] == precision] for label in "QK"},
              "Training tokens (B)", "Linear CE minus variant CE")
    chart(plots / "precision-gap.svg", "Same checkpoint: FP8 minus BF16 inference CE",
          {label: [(r["step"] * 2097152 / 1e9, r["mean_delta_ce"]) for r in effects if r["comparison"] == f"{label}_fp8_minus_bf16"] for label in "LQK"},
          "Training tokens (B)", "FP8 CE minus BF16 CE")
    for metric in ("normalized_entropy", "mean_distance", "logit_std", "max_probability", "distance_over1024_mass"):
        chart(plots / f"attention-{metric}.svg", f"Final checkpoint BF16: {metric}",
              {label: [(r["layer"] + 1, r["mean"]) for r in attention_rows if r["variant"] == label and r["precision"] == "bf16" and r["step"] == max(steps) and r["metric"] == metric] for label in "LQK"},
              "Layer (1-based; inspect window_left in raw rows)", metric)
    links = "\n".join(f"- [{p.stem}](plots/{p.name})" for p in sorted(plots.glob("*.svg")))
    final = [r for r in summaries if r["step"] == max(steps)]
    table = "\n".join(f"|{r['variant']}|{r['precision']}|{r['sequence_macro_ce']:.6f}|" for r in final)
    (root / "report.md").write_text(
        "# Q/K checkpoint analysis\n\n"
        "L=all-linear, Q=Q-only SiLU, K=K-only SiLU. All trained with FP8.\n\n"
        f"Matched {len(samples)} held-out sequences, fixed sampled scored positions; {len(steps)} checkpoint stages.\n\n"
        "|Variant|Inference|Final sampled CE|\n|---|---|---:|\n" + table + "\n\n"
        "## Interpretation boundaries\n\n"
        "CE is an equal-sequence average on a source-balanced sample, not official full validation CE or BPB. "
        "Paired bootstrap intervals describe sequence-sampling uncertainty, not training-seed uncertainty. "
        "The input has full8K context with right padding; only label-valid next-token positions are scored. "
        "Attention statistics use post-norm/post-RoPE Q/K, actual causal/window masks, FP32 reconstruction "
        "with TF32 disabled; reconstructed probabilities are not bit-exact FlashAttention internals. "
        "Long-range mass is constrained by each layer's window; lower entropy is not automatically better. "
        "FP8/BF16 comparisons change forward GEMMs of the SAME checkpoint using its saved Float8 recipe; "
        "they do not undo FP8 training or isolate input quantization from accumulation error. "
        "No channel-wise semantic alignment across models and no causal superiority claim.\n\n"
        "## Outputs\n\nsummary.csv; per_sequence_ce.csv; paired_effects.csv; attention_summary.csv. "
        "Raw per-token/head statistics and small binned attention snapshots remain in rank-* directories.\n\n"
        "## Figures\n\n" + links + "\n")
    return summaries, effects


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    build_report(parser.parse_args().directory)
