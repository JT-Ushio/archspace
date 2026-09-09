#!/usr/bin/env python3
"""Compare the Down SiLU/RMSNorm precision ablations stored in W&B.

With no run arguments this selects the latest runs whose display names match the
two BF16 launcher defaults. Run identifiers may also be W&B URLs,
``entity/project/run_id`` paths, run IDs, or display names.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


DEFAULT_BF16_SILU = "olmo3-1b-precision-down-silu-bf16"
DEFAULT_BF16_RMS = "olmo3-1b-precision-down-rms-bf16"
CHECKPOINT_FRACTIONS = (0.25, 0.50, 0.75, 1.00)
WINDOW_FRACTIONS = (0.10, 0.05)
POINT_WINDOW_FRACTION = 0.01

LOSS_KEY_CANDIDATES = (
    "train/CE loss",
    "train/ce_loss",
    "train/cross_entropy_loss",
    "train/cross entropy loss",
    "train/total loss",
    "train/loss",
    "loss/train",
    "loss",
)

STEP_KEY_CANDIDATES = (
    "trainer/global_step",
    "train/global_step",
    "global_step",
    "trainer/step",
    "train/step",
    "step",
    "_step",
)


@dataclass
class RunSeries:
    label: str
    run: Any
    loss_key: str
    step_key: str
    points: list[tuple[float, float]]

    @property
    def max_step(self) -> float:
        return self.points[-1][0]

    @property
    def normalized_points(self) -> list[tuple[float, float]]:
        maximum = self.max_step
        if maximum <= 0:
            raise ValueError(f"{self.label}: maximum step must be positive")
        return [(step / maximum, loss) for step, loss in self.points]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Down-SiLU and Down-RMSNorm W&B curves. Positive deltas "
            "mean RMSNorm has lower loss."
        )
    )
    parser.add_argument("--entity", default="archspace")
    parser.add_argument("--project", default="MoSE")
    parser.add_argument("--bf16-silu-run", default=DEFAULT_BF16_SILU)
    parser.add_argument("--bf16-rms-run", default=DEFAULT_BF16_RMS)
    parser.add_argument(
        "--fp8-silu-run",
        help="Optional FP8 SiLU run URL/path/ID/display name.",
    )
    parser.add_argument(
        "--fp8-rms-run",
        help="Optional FP8 RMSNorm run URL/path/ID/display name.",
    )
    parser.add_argument(
        "--loss-key",
        help="Override automatic loss metric detection, e.g. 'train/CE loss'.",
    )
    parser.add_argument(
        "--step-key",
        help="Override automatic optimizer-step metric detection.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("precision_comparison"),
        help="Directory for CSV files and the optional plot.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Do not create loss_comparison.png.",
    )
    parser.add_argument(
        "--grid-points",
        type=int,
        default=401,
        help="Number of normalized-progress points used for AUC and CSV output.",
    )
    return parser.parse_args()


def wandb_path_from_url(spec: str) -> str | None:
    match = re.search(
        r"wandb\.ai/([^/]+)/([^/]+)/(?:runs|sweeps)/([^/?#]+)", spec
    )
    if match is None:
        return None
    return "/".join(match.groups())


def resolve_run(api: Any, spec: str, entity: str, project: str) -> Any:
    """Resolve a URL/path/ID/name and prefer the latest exact name match."""
    url_path = wandb_path_from_url(spec)
    if url_path is not None:
        return api.run(url_path)

    clean_spec = spec.strip().strip("/")
    if clean_spec.count("/") == 2:
        return api.run(clean_spec)

    matches = list(
        api.runs(
            f"{entity}/{project}",
            filters={"display_name": clean_spec},
            order="-created_at",
            per_page=50,
        )
    )
    if matches:
        selected = matches[0]
        if len(matches) > 1:
            print(
                f"[resolve] {len(matches)} runs are named {clean_spec!r}; "
                f"using latest run id={selected.id}, created_at={selected.created_at}",
                file=sys.stderr,
            )
        return selected

    # If there was no display-name match, treat the value as a run ID.
    return api.run(f"{entity}/{project}/{clean_spec}")


def history_sample(run: Any) -> list[dict[str, Any]]:
    sample = run.history(samples=200, pandas=False)
    if isinstance(sample, list):
        return [row for row in sample if isinstance(row, dict)]
    if hasattr(sample, "to_dict"):
        return list(sample.to_dict(orient="records"))
    return []


def available_keys(run: Any) -> set[str]:
    keys = {"_step"}
    try:
        summary = run.summary._json_dict
    except AttributeError:
        summary = dict(run.summary)
    keys.update(str(key) for key in summary)
    for row in history_sample(run):
        keys.update(str(key) for key in row)
    return keys


def loss_key_score(key: str) -> int:
    lowered = key.lower().replace("_", " ").replace("-", " ")
    if "loss" not in lowered:
        return -10_000
    score = 10
    if lowered.startswith("train/") or "train" in lowered:
        score += 50
    if "ce loss" in lowered or "cross entropy" in lowered:
        score += 40
    if "total loss" in lowered:
        score += 20
    for unwanted in ("eval", "validation", "downstream", "z loss", "aux", "grad"):
        if unwanted in lowered:
            score -= 80
    return score


def choose_common_key(
    all_keys: Sequence[set[str]], explicit: str | None, candidates: Sequence[str], kind: str
) -> str:
    if explicit is not None:
        return explicit
    common = set.intersection(*all_keys)
    for candidate in candidates:
        if candidate in common:
            return candidate
    if kind == "loss":
        ranked = sorted(common, key=lambda key: (loss_key_score(key), key), reverse=True)
        if ranked and loss_key_score(ranked[0]) > 0:
            return ranked[0]
    available = ", ".join(sorted(common))
    raise RuntimeError(
        f"Could not automatically detect a common {kind} key. "
        f"Pass --{kind}-key explicitly. Common keys: {available}"
    )


def is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def fetch_points(run: Any, step_key: str, loss_key: str) -> list[tuple[float, float]]:
    deduplicated: dict[float, float] = {}
    for row in run.scan_history(keys=[step_key, loss_key], page_size=1000):
        step = row.get(step_key)
        loss = row.get(loss_key)
        if is_finite_number(step) and is_finite_number(loss):
            deduplicated[float(step)] = float(loss)
    points = sorted(deduplicated.items())
    if len(points) < 2:
        raise RuntimeError(
            f"Run {run.id} yielded only {len(points)} usable rows for "
            f"step={step_key!r}, loss={loss_key!r}."
        )
    return points


def interpolate(points: Sequence[tuple[float, float]], x: float) -> float:
    if x <= points[0][0]:
        return points[0][1]
    if x >= points[-1][0]:
        return points[-1][1]
    left = 0
    right = len(points) - 1
    while left + 1 < right:
        middle = (left + right) // 2
        if points[middle][0] <= x:
            left = middle
        else:
            right = middle
    x0, y0 = points[left]
    x1, y1 = points[right]
    if x1 == x0:
        return y1
    weight = (x - x0) / (x1 - x0)
    return y0 + weight * (y1 - y0)


def trapezoid_mean(
    points: Sequence[tuple[float, float]], start: float, end: float
) -> float:
    if end <= start:
        raise ValueError("Integration interval must have positive length")
    xs = [start]
    xs.extend(x for x, _ in points if start < x < end)
    xs.append(end)
    area = 0.0
    for x0, x1 in zip(xs, xs[1:]):
        y0 = interpolate(points, x0)
        y1 = interpolate(points, x1)
        area += 0.5 * (y0 + y1) * (x1 - x0)
    return area / (end - start)


def nearest_raw_step(series: RunSeries, progress: float) -> float:
    target = progress * series.max_step
    return min(series.points, key=lambda point: abs(point[0] - target))[0]


def local_progress_mean(
    points: Sequence[tuple[float, float]], progress: float
) -> float:
    """Average a clipped 1%-wide window; anomalous values remain included."""
    half_width = POINT_WINDOW_FRACTION / 2.0
    start = max(points[0][0], progress - half_width)
    end = min(1.0, progress + half_width)
    if end <= start:
        return interpolate(points, progress)
    return trapezoid_mean(points, start, end)


def build_pair_rows(
    precision: str,
    silu: RunSeries,
    rms: RunSeries,
    grid_points: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if silu.max_step != rms.max_step:
        raise ValueError(
            f"{precision}: unequal last steps ({silu.max_step}, {rms.max_step}); "
            "refusing to compare different training lengths by normalized progress."
        )
    for series in (silu, rms):
        suspicious = [step for step, loss in series.points if loss <= 0]
        if suspicious:
            print(
                f"WARNING: {series.label} has non-positive loss at steps {suspicious}. "
                "Values are retained; affected window means/AUC need investigation.",
                file=sys.stderr,
            )
    silu_norm = silu.normalized_points
    rms_norm = rms.normalized_points
    common_start = max(silu_norm[0][0], rms_norm[0][0])
    if common_start >= 1.0:
        raise RuntimeError(f"{precision}: no overlapping training progress")

    aligned: list[dict[str, Any]] = []
    for index in range(grid_points):
        progress = common_start + (1.0 - common_start) * index / (grid_points - 1)
        silu_loss = interpolate(silu_norm, progress)
        rms_loss = interpolate(rms_norm, progress)
        aligned.append(
            {
                "precision": precision,
                "progress": progress,
                "progress_percent": progress * 100.0,
                "silu_loss": silu_loss,
                "rms_loss": rms_loss,
                "delta_silu_minus_rms": silu_loss - rms_loss,
            }
        )

    checkpoint_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for fraction in CHECKPOINT_FRACTIONS:
        silu_loss = local_progress_mean(silu_norm, fraction)
        rms_loss = local_progress_mean(rms_norm, fraction)
        row = {
            "precision": precision,
            "metric": f"progress_{int(fraction * 100)}pct_window_mean",
            "progress": fraction,
            "silu_nearest_step": nearest_raw_step(silu, fraction),
            "rms_nearest_step": nearest_raw_step(rms, fraction),
            "silu_loss": silu_loss,
            "rms_loss": rms_loss,
            "delta_silu_minus_rms": silu_loss - rms_loss,
        }
        checkpoint_rows.append(row)
        summary_rows.append(row.copy())

    for window in WINDOW_FRACTIONS:
        start = 1.0 - window
        silu_mean = trapezoid_mean(silu_norm, start, 1.0)
        rms_mean = trapezoid_mean(rms_norm, start, 1.0)
        summary_rows.append(
            {
                "precision": precision,
                "metric": f"last_{int(window * 100)}pct_mean",
                "progress": "",
                "silu_nearest_step": "",
                "rms_nearest_step": "",
                "silu_loss": silu_mean,
                "rms_loss": rms_mean,
                "delta_silu_minus_rms": silu_mean - rms_mean,
            }
        )

    silu_auc = trapezoid_mean(silu_norm, common_start, 1.0)
    rms_auc = trapezoid_mean(rms_norm, common_start, 1.0)
    summary_rows.append(
        {
            "precision": precision,
            "metric": "full_curve_auc_mean",
            "progress": "",
            "silu_nearest_step": "",
            "rms_nearest_step": "",
            "silu_loss": silu_auc,
            "rms_loss": rms_auc,
            "delta_silu_minus_rms": silu_auc - rms_auc,
        }
    )
    return aligned, checkpoint_rows, summary_rows


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_run_metadata(series: RunSeries) -> None:
    run = series.run
    print(
        f"  {series.label:<10} id={run.id} state={run.state} "
        f"name={run.name!r} rows={len(series.points)} max_step={series.max_step:g}"
    )


def print_pair_summary(precision: str, rows: Sequence[dict[str, Any]]) -> None:
    print(f"\n{precision.upper()} (delta = SiLU loss - RMS loss; positive favors RMSNorm)")
    print(f"  {'metric':<24} {'SiLU':>12} {'RMSNorm':>12} {'delta':>12}")
    for row in rows:
        print(
            f"  {row['metric']:<24} "
            f"{float(row['silu_loss']):>12.6f} "
            f"{float(row['rms_loss']):>12.6f} "
            f"{float(row['delta_silu_minus_rms']):>+12.6f}"
        )


def maybe_plot(output_path: Path, pairs: dict[str, tuple[RunSeries, RunSeries]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n[plot] matplotlib is unavailable; CSV output was still written.")
        return

    _, axes = plt.subplots(1, len(pairs), figsize=(7 * len(pairs), 5), squeeze=False)
    for axis, (precision, (silu, rms)) in zip(axes[0], pairs.items()):
        for series, label in ((silu, "Down-SiLU"), (rms, "Down-RMSNorm")):
            normalized = series.normalized_points
            axis.plot(
                [progress * 100.0 for progress, _ in normalized],
                [loss for _, loss in normalized],
                label=label,
                linewidth=1.3,
            )
        axis.set_title(precision.upper())
        axis.set_xlabel("Training progress (%)")
        axis.set_ylabel(silu.loss_key)
        axis.grid(alpha=0.25)
        axis.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    print(f"[plot] wrote {output_path}")


def main() -> int:
    args = parse_args()
    if args.grid_points < 2:
        raise ValueError("--grid-points must be at least 2")
    if bool(args.fp8_silu_run) != bool(args.fp8_rms_run):
        raise ValueError("Pass both --fp8-silu-run and --fp8-rms-run, or neither")

    try:
        import wandb
    except ImportError as error:
        raise RuntimeError(
            "The wandb package is required. Run this script in the OLMo environment."
        ) from error

    api = wandb.Api(timeout=60)
    specs: list[tuple[str, str]] = [
        ("bf16_silu", args.bf16_silu_run),
        ("bf16_rms", args.bf16_rms_run),
    ]
    if args.fp8_silu_run and args.fp8_rms_run:
        specs.extend(
            (("fp8_silu", args.fp8_silu_run), ("fp8_rms", args.fp8_rms_run))
        )

    resolved = {
        label: resolve_run(api, spec, args.entity, args.project) for label, spec in specs
    }
    keys = {label: available_keys(run) for label, run in resolved.items()}
    loss_key = choose_common_key(
        list(keys.values()), args.loss_key, LOSS_KEY_CANDIDATES, "loss"
    )
    step_key = choose_common_key(
        list(keys.values()), args.step_key, STEP_KEY_CANDIDATES, "step"
    )

    series = {
        label: RunSeries(
            label=label,
            run=run,
            loss_key=loss_key,
            step_key=step_key,
            points=fetch_points(run, step_key, loss_key),
        )
        for label, run in resolved.items()
    }

    print(f"Detected loss key: {loss_key!r}")
    print(f"Detected step key: {step_key!r}")
    print("Resolved runs:")
    for item in series.values():
        print_run_metadata(item)

    pairs: dict[str, tuple[RunSeries, RunSeries]] = {
        "bf16": (series["bf16_silu"], series["bf16_rms"])
    }
    if "fp8_silu" in series:
        pairs["fp8"] = (series["fp8_silu"], series["fp8_rms"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for item in series.values():
        write_csv(
            args.output_dir / f"{item.label}_raw.csv",
            (
                {
                    "step": step,
                    "progress": step / item.max_step,
                    "loss": loss,
                }
                for step, loss in item.points
            ),
            ("step", "progress", "loss"),
        )
    all_summary: list[dict[str, Any]] = []
    all_checkpoints: list[dict[str, Any]] = []
    for precision, (silu, rms) in pairs.items():
        aligned, checkpoints, summary = build_pair_rows(
            precision, silu, rms, args.grid_points
        )
        write_csv(
            args.output_dir / f"{precision}_aligned.csv",
            aligned,
            (
                "precision",
                "progress",
                "progress_percent",
                "silu_loss",
                "rms_loss",
                "delta_silu_minus_rms",
            ),
        )
        all_checkpoints.extend(checkpoints)
        all_summary.extend(summary)
        print_pair_summary(precision, summary)

    common_fields = (
        "precision",
        "metric",
        "progress",
        "silu_nearest_step",
        "rms_nearest_step",
        "silu_loss",
        "rms_loss",
        "delta_silu_minus_rms",
    )
    write_csv(args.output_dir / "checkpoint_comparison.csv", all_checkpoints, common_fields)
    write_csv(args.output_dir / "summary.csv", all_summary, common_fields)

    if "fp8" in pairs:
        summary_by_pair = {
            (row["precision"], row["metric"]): row for row in all_summary
        }
        print("\nINTERACTION (FP8 delta - BF16 delta; positive is extra FP8 RMS benefit)")
        print(f"  {'metric':<24} {'interaction':>12}")
        for metric in [row["metric"] for row in all_summary if row["precision"] == "bf16"]:
            bf16_delta = float(
                summary_by_pair[("bf16", metric)]["delta_silu_minus_rms"]
            )
            fp8_delta = float(
                summary_by_pair[("fp8", metric)]["delta_silu_minus_rms"]
            )
            print(f"  {metric:<24} {fp8_delta - bf16_delta:>+12.6f}")

    metadata_rows = []
    for item in series.values():
        metadata_rows.append(
            {
                "label": item.label,
                "run_id": item.run.id,
                "run_name": item.run.name,
                "run_path": "/".join(item.run.path),
                "state": item.run.state,
                "created_at": item.run.created_at,
                "loss_key": item.loss_key,
                "step_key": item.step_key,
                "history_rows": len(item.points),
                "max_step": item.max_step,
            }
        )
    write_csv(
        args.output_dir / "runs.csv",
        metadata_rows,
        (
            "label",
            "run_id",
            "run_name",
            "run_path",
            "state",
            "created_at",
            "loss_key",
            "step_key",
            "history_rows",
            "max_step",
        ),
    )

    if not args.no_plot:
        maybe_plot(args.output_dir / "loss_comparison.png", pairs)
    print(f"\nWrote comparison files to {args.output_dir.resolve()}")
    print("These are descriptive curve comparisons; one seed is not a significance test.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
