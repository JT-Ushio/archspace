#!/usr/bin/env python3
"""Run the OLMo 3 base-easy suite as independent one-GPU OLMES shards.

OLMES' built-in multi-worker path assigns CUDA_VISIBLE_DEVICES inside a worker
function. With Python's spawn mode, CUDA-capable modules are imported before
that function runs, which can place every worker on physical GPU 0. This
driver assigns CUDA_VISIBLE_DEVICES before starting each Python interpreter,
so eight processes are eight genuinely independent data-parallel workers.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


EASY_SUITES = (
    "olmo3:base_easy:code_bpb",
    "olmo3:base_easy:math_bpb",
    "olmo3:base_easy:qa_rc",
    "olmo3:base_easy:qa_bpb",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--olmes-repo", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--min-batch-size", type=int, default=1)
    parser.add_argument("--devices", help="Comma-separated allocated CUDA IDs or UUIDs")
    parser.add_argument("--assignment", choices=("balanced", "round-robin"), default="balanced")
    parser.add_argument(
        "--timings", type=Path, default=Path(__file__).with_name("historical_timings.json")
    )
    parser.add_argument(
        "--plan-only", action="store_true", help="Write specs without GPU inference"
    )
    parser.add_argument("--oom-retry", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gpu-sample-seconds", type=float, default=15)
    return parser.parse_args()


def task_alias(task: dict[str, Any]) -> str:
    return task.get("metadata", {}).get("alias") or task["task_name"]


def load_timings(path: Path) -> dict[str, float]:
    profile = json.loads(path.read_text(encoding="utf-8"))
    costs = profile["task_wall_seconds"]
    if any(
        isinstance(x, bool) or not math.isfinite(float(x)) or float(x) <= 0 for x in costs.values()
    ):
        raise ValueError("Task wall times must be finite positive numbers")
    return {name: float(value) for name, value in costs.items()}


def assign_tasks(task_configs, num_shards, costs, assignment="balanced"):
    """Longest tasks first, preserving original global IDs for result collation."""
    if num_shards < 1 or num_shards > len(task_configs):
        raise ValueError("Each GPU worker must have at least one task")
    aliases = [task_alias(task) for task in task_configs]
    if len(aliases) != len(set(aliases)):
        raise ValueError("Duplicate task aliases; refuse an ambiguous result mapping")
    fallback = statistics.median(costs.values()) if costs else 1.0
    estimated = [costs.get(alias, fallback) for alias in aliases]
    groups = [[] for _ in range(num_shards)]
    totals = [0.0] * num_shards
    order = (
        sorted(range(len(task_configs)), key=lambda i: (-estimated[i], i))
        if assignment == "balanced"
        else range(len(task_configs))
    )
    for global_idx in order:
        shard_idx = (
            min(range(num_shards), key=lambda i: (totals[i], len(groups[i]), i))
            if assignment == "balanced"
            else global_idx % num_shards
        )
        groups[shard_idx].append((global_idx, task_configs[global_idx]))
        totals[shard_idx] += estimated[global_idx]
    return groups, totals, estimated


def allocated_devices(num_shards: int, explicit: str | None = None) -> list[str]:
    visible = explicit if explicit is not None else os.environ.get("CUDA_VISIBLE_DEVICES")
    devices = (
        [x.strip() for x in visible.split(",")]
        if visible is not None
        else [str(i) for i in range(num_shards)]
    )
    if any(not x or x == "-1" for x in devices) or len(set(devices)) != len(devices):
        raise ValueError("CUDA device IDs must be nonempty, unique, and enabled")
    if len(devices) < num_shards:
        raise ValueError(f"Only {len(devices)} allocated GPUs for {num_shards} workers")
    return devices[:num_shards]


def resolve_suite(name: str, suite_configs: dict[str, dict[str, Any]]) -> list[str]:
    if name not in suite_configs:
        return [name]
    resolved: list[str] = []
    for child in suite_configs[name]["tasks"]:
        resolved.extend(resolve_suite(child, suite_configs))
    return resolved


def build_task_configs() -> list[dict[str, Any]]:
    from oe_eval.configs.task_suites import TASK_SUITE_CONFIGS
    from oe_eval.configs.tasks import TASK_CONFIGS

    task_names: list[str] = []
    for suite in EASY_SUITES:
        task_names.extend(resolve_suite(suite, TASK_SUITE_CONFIGS))

    task_configs: list[dict[str, Any]] = []
    for task_name in task_names:
        if task_name in TASK_CONFIGS:
            task_config = copy.deepcopy(TASK_CONFIGS[task_name])
            task_config.setdefault("metadata", {})["alias"] = task_name
        else:
            task_config = {"task_name": task_name}
        task_configs.append(task_config)
    return task_configs


def write_specs(
    task_configs: list[dict[str, Any]],
    output_dir: Path,
    num_shards: int,
    costs: dict[str, float] | None = None,
    assignment: str = "balanced",
) -> list[Path]:
    specs_dir = output_dir / "task_specs"
    specs_dir.mkdir(parents=True, exist_ok=True)
    costs = costs or {}
    shard_tasks, totals, estimated = assign_tasks(task_configs, num_shards, costs, assignment)
    _, round_robin_totals, _ = assign_tasks(task_configs, num_shards, costs, "round-robin")

    specs: list[Path] = []
    manifest_tasks: list[dict[str, Any]] = []
    for shard_idx, items in enumerate(shard_tasks):
        spec_path = specs_dir / f"shard-{shard_idx:02d}.jsonl"
        with spec_path.open("w", encoding="utf-8") as stream:
            for local_idx, (global_idx, task_config) in enumerate(items):
                stream.write(json.dumps(task_config, ensure_ascii=False) + "\n")
                manifest_tasks.append(
                    {
                        "global_idx": global_idx,
                        "shard_idx": shard_idx,
                        "local_idx": local_idx,
                        "task_name": task_config.get("task_name"),
                        "alias": task_config.get("metadata", {}).get("alias"),
                        "estimated_wall_seconds": estimated[global_idx],
                    }
                )
        specs.append(spec_path)

    manifest_tasks.sort(key=lambda item: item["global_idx"])
    manifest = {
        "suites": list(EASY_SUITES),
        "num_tasks": len(task_configs),
        "num_shards": num_shards,
        "tasks": manifest_tasks,
        "assignment": assignment,
        "estimated_worker_seconds": totals,
        "round_robin_estimated_worker_seconds": round_robin_totals,
        "timing_coverage": sum(task_alias(task) in costs for task in task_configs),
    }
    with (output_dir / "shard_manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
    return specs


def count_completed(output_dir: Path) -> int:
    return sum(1 for _ in (output_dir / "shards").glob("shard-*/task-*-metrics.json"))


def prepare_scoring_model(model_dir: Path, output_dir: Path) -> Path:
    """Read-only weight view with a scoring config; OLMES rejects unknown model args."""
    scoring_dir = output_dir / "scoring_model"
    scoring_dir.mkdir()
    config = json.loads((model_dir / "config.json").read_text())
    config["use_cache"] = False
    for path in model_dir.iterdir():
        if path.name != "config.json" and path.is_file():
            (scoring_dir / path.name).symlink_to(path.resolve())
    (scoring_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    return scoring_dir


def worker_command(args, spec_path, shard_output, batch_size):
    return [
        sys.executable,
        "-m",
        "oe_eval.launch",
        "--model",
        str(getattr(args, "scoring_model", args.model)),
        "--model-type",
        "hf",
        "--model-args",
        '{"trust_remote_code":true,"dtype":"bfloat16"}',
        "--task",
        str(spec_path),
        "--output-dir",
        str(shard_output),
        "--gpus",
        "1",
        "--num-workers",
        "1",
        "--batch-size",
        str(batch_size),
        "--save-raw-requests",
        "false",
    ]


def worker_environment(device: str, worker_idx: int) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = device
    # Per-process imports avoid contention when HF copies trust_remote_code modules.
    env["HF_MODULES_CACHE"] = str(
        Path(env.get("TMPDIR", "/tmp")) / f"olmes-worker-{worker_idx}" / "hf_modules"
    )
    return env


def sample_gpus(output_dir: Path) -> None:
    if not shutil.which("nvidia-smi"):
        return
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total,power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if result.returncode == 0:
            with (output_dir / "gpu_samples.jsonl").open("a", encoding="utf-8") as stream:
                json.dump(
                    {
                        "time": datetime.now().astimezone().isoformat(),
                        "devices": list(csv.reader(result.stdout.splitlines())),
                    },
                    stream,
                )
                stream.write("\n")
    except (OSError, subprocess.TimeoutExpired):
        pass  # Missing telemetry does not invalidate evaluation results.


def launch_shards(args: argparse.Namespace, specs: list[Path], num_tasks: int) -> list[dict]:
    shards_dir = args.output_dir / "shards"
    logs_dir = args.output_dir / "logs"
    shards_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    devices = allocated_devices(args.num_shards, args.devices)
    workers = []

    def start(worker):
        shard_idx, spec_path = worker["shard_idx"], worker["spec"]
        shard_output = shards_dir / f"shard-{shard_idx:02d}"
        shard_output.mkdir(parents=True, exist_ok=True)
        log_path = logs_dir / f"shard-{shard_idx:02d}.log"
        worker["log_offset"] = log_path.stat().st_size if log_path.exists() else 0
        log_stream = log_path.open("ab", buffering=0)
        worker["log_stream"] = log_stream
        env = worker_environment(devices[shard_idx], shard_idx)
        command = worker_command(args, spec_path, shard_output, worker["batch_size"])
        process = subprocess.Popen(
            command,
            cwd=args.olmes_repo,
            env=env,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        worker.update(process=process, done=False)
        print(
            f"Started shard {shard_idx:02d} on allocated GPU {devices[shard_idx]}: "
            f"pid={process.pid}, batch={worker['batch_size']}",
            flush=True,
        )

    started = time.monotonic()
    next_sample, last_count = started, -1
    try:
        for shard_idx, spec_path in enumerate(specs):
            worker = {
                "shard_idx": shard_idx,
                "spec": spec_path,
                "batch_size": args.batch_size,
                "attempts": [],
                "done": False,
            }
            workers.append(worker)
            start(worker)
        while not all(worker["done"] for worker in workers):
            for worker in workers:
                if worker["done"] or worker["process"].poll() is None:
                    continue
                code = worker["process"].wait()
                worker["log_stream"].close()
                worker["attempts"].append({"batch_size": worker["batch_size"], "exit_code": code})
                if code:
                    with (logs_dir / f"shard-{worker['shard_idx']:02d}.log").open("rb") as stream:
                        stream.seek(worker["log_offset"])
                        tail = stream.read().decode("utf-8", errors="replace")
                    is_oom = "CUDA out of memory" in tail or "torch.OutOfMemoryError" in tail
                    if args.oom_retry and is_oom and worker["batch_size"] > args.min_batch_size:
                        worker["batch_size"] = max(args.min_batch_size, worker["batch_size"] // 2)
                        print(
                            f"Shard {worker['shard_idx']}: CUDA OOM; retry batch="
                            f"{worker['batch_size']} with completed tasks retained",
                            flush=True,
                        )
                        start(worker)
                        continue
                    raise RuntimeError(
                        f"Shard {worker['shard_idx']} failed with exit code {code}; see {logs_dir}"
                    )
                worker["done"] = True
                worker["wall_seconds"] = round(time.monotonic() - started, 3)
            completed = count_completed(args.output_dir)
            if completed != last_count:
                states = " ".join(
                    f"{w['shard_idx']}:{'DONE' if w['done'] else 'RUN'}" for w in workers
                )
                print(f"Progress: {completed}/{num_tasks}; shards [{states}]", flush=True)
                last_count = completed
            now = time.monotonic()
            if args.gpu_sample_seconds > 0 and now >= next_sample:
                sample_gpus(args.output_dir)
                next_sample = now + args.gpu_sample_seconds
            if not all(worker["done"] for worker in workers):
                time.sleep(2)
    finally:
        for worker in workers:
            process = worker.get("process")
            if process is not None and process.poll() is None:
                # Kill the launcher's child run_eval too, confined to this worker process group.
                import signal

                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            if worker.get("log_stream"):
                worker["log_stream"].close()

    return [
        {key: worker[key] for key in ("shard_idx", "batch_size", "attempts", "wall_seconds")}
        for worker in workers
    ]


def merge_results(
    output_dir: Path,
    num_shards: int,
    num_tasks: int,
    batch_size: int,
    workers: list[dict] | None = None,
) -> None:
    manifest = json.loads((output_dir / "shard_manifest.json").read_text(encoding="utf-8"))
    index_by_location = {
        (task["shard_idx"], task["local_idx"]): task["global_idx"] for task in manifest["tasks"]
    }

    all_metrics: list[dict[str, Any]] = []
    all_tasks: list[tuple[int, dict[str, Any]]] = []
    all_primary_scores: list[tuple[int, str]] = []
    alias_by_index = {task["global_idx"]: task["alias"] for task in manifest["tasks"]}
    model_config: dict[str, Any] | None = None
    for shard_idx in range(num_shards):
        shard_dir = output_dir / "shards" / f"shard-{shard_idx:02d}"
        metrics_all_path = shard_dir / "metrics-all.jsonl"
        summary_path = shard_dir / "metrics.json"
        if not metrics_all_path.is_file() or not summary_path.is_file():
            raise RuntimeError(f"Missing completed result for shard {shard_idx}: {shard_dir}")

        with metrics_all_path.open("r", encoding="utf-8") as stream:
            for line in stream:
                metric = json.loads(line)
                local_idx = int(metric["task_idx"])
                global_idx = index_by_location[(shard_idx, local_idx)]
                metric["shard_idx"] = shard_idx
                metric["shard_task_idx"] = local_idx
                metric["task_idx"] = global_idx
                all_metrics.append(metric)

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if model_config is None:
            model_config = summary.get("model_config", {})
        for local_idx, task in enumerate(summary.get("tasks", [])):
            global_idx = index_by_location[(shard_idx, local_idx)]
            if task.get("alias") != alias_by_index[global_idx]:
                raise RuntimeError(f"Task alias mismatch for original task {global_idx}")
            all_tasks.append((global_idx, task))
        for local_idx, score in enumerate(summary.get("all_primary_scores", [])):
            global_idx = index_by_location[(shard_idx, local_idx)]
            all_primary_scores.append((global_idx, score))

    all_metrics.sort(key=lambda item: item["task_idx"])
    all_tasks.sort(key=lambda item: item[0])
    all_primary_scores.sort(key=lambda item: item[0])
    expected = set(range(num_tasks))
    if (
        len(all_metrics) != num_tasks
        or len(all_tasks) != num_tasks
        or len(all_primary_scores) != num_tasks
        or {row["task_idx"] for row in all_metrics} != expected
        or {idx for idx, _ in all_tasks} != expected
    ):
        raise RuntimeError(
            f"Merged task count mismatch: metrics={len(all_metrics)}, tasks={len(all_tasks)}, expected={num_tasks}"
        )

    with (output_dir / "metrics-all.jsonl").open("w", encoding="utf-8") as stream:
        for metric in all_metrics:
            stream.write(json.dumps(metric, ensure_ascii=False) + "\n")
    merged_summary = {
        "all_primary_scores": [score for _, score in all_primary_scores],
        "tasks": [task for _, task in all_tasks],
        "model_config": model_config or {},
        "parallelism": {"data_parallel": num_shards, "tensor_parallel": 1},
        "batch_size_per_gpu": batch_size,
        "worker_batch_sizes": {str(w["shard_idx"]): w["batch_size"] for w in workers or []},
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(merged_summary, stream, ensure_ascii=False)
    completion = {
        "status": "complete",
        "num_tasks": num_tasks,
        "num_shards": num_shards,
        "batch_size_per_gpu": batch_size,
        "metrics_all": str(output_dir / "metrics-all.jsonl"),
        "metrics_summary": str(output_dir / "metrics.json"),
        "worker_batch_sizes": merged_summary["worker_batch_sizes"],
    }
    with (output_dir / "evaluation_complete.json").open("w", encoding="utf-8") as stream:
        json.dump(completion, stream, ensure_ascii=False, indent=2)


def collect_task_timings(output_dir: Path) -> dict:
    """Use logged wall intervals; OLMES processing_time omits inference."""
    manifest = json.loads((output_dir / "shard_manifest.json").read_text())
    pattern = re.compile(r"(\d{4}-\d{2}-\d{2})[ :](\d{2}:\d{2}:\d{2}),(\d+)")
    costs = {}
    for shard_idx in range(manifest["num_shards"]):
        tasks = sorted(
            (task for task in manifest["tasks"] if task["shard_idx"] == shard_idx),
            key=lambda task: task["local_idx"],
        )
        events = []
        for line in (
            (output_dir / "logs" / f"shard-{shard_idx:02d}.log")
            .read_text(errors="replace")
            .splitlines()
        ):
            match = pattern.match(line)
            if match and ("**** Processing task:" in line or "Saving final metrics in" in line):
                events.append(datetime.fromisoformat(f"{match[1]}T{match[2]}.{match[3]}"))
        # Retries can repeat tasks. Do not turn ambiguous log intervals into timing estimates.
        if len(events) == len(tasks) + 1:
            for i, task in enumerate(tasks):
                elapsed = (events[i + 1] - events[i]).total_seconds()
                if elapsed > 0:
                    costs[task["alias"]] = elapsed
    return {
        "timing_basis": "Logged task wall intervals, including inference",
        "source": str(output_dir),
        "task_wall_seconds": costs,
    }


def main() -> None:
    args = parse_args()
    if args.num_shards < 1 or not 1 <= args.min_batch_size <= args.batch_size:
        raise ValueError("Workers and batch sizes must be positive; min batch cannot exceed batch")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Use a fresh output directory; existing evaluation results are preserved")
    if not args.model.is_dir() or not (args.model / "config.json").is_file():
        raise ValueError("--model must be a local HF directory containing config.json")
    if not (
        (args.model / "model.safetensors").is_file()
        or (args.model / "model.safetensors.index.json").is_file()
    ):
        raise ValueError("Missing HF safetensors weights or sharded index")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.olmes_repo))
    task_configs = build_task_configs()
    costs = load_timings(args.timings)
    print(
        f"Resolved {len(task_configs)} easy tasks into {args.num_shards} one-GPU shards "
        f"at batch size {args.batch_size}.",
        flush=True,
    )
    specs = write_specs(task_configs, args.output_dir, args.num_shards, costs, args.assignment)
    manifest = json.loads((args.output_dir / "shard_manifest.json").read_text())
    print(
        f"Task wall estimates: selected={manifest['estimated_worker_seconds']}; "
        f"round-robin={manifest['round_robin_estimated_worker_seconds']}",
        flush=True,
    )
    run_config = {
        "model": str(args.model.resolve()),
        "olmes_repo": str(args.olmes_repo.resolve()),
        "suites": list(EASY_SUITES),
        "assignment": args.assignment,
        "num_workers": args.num_shards,
        "tensor_parallel": 1,
        "batch_size_per_gpu": args.batch_size,
        "min_batch_size": args.min_batch_size,
        "dtype": "bfloat16",
        "use_cache": False,
        "plan_only": args.plan_only,
        "timings": str(args.timings.resolve()),
    }
    (args.output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2) + "\n")
    if args.plan_only:
        print("Plan written; no model forward or GPU worker started", flush=True)
        return
    args.scoring_model = prepare_scoring_model(args.model, args.output_dir)
    run_config["scoring_model"] = str(args.scoring_model)
    (args.output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2) + "\n")
    started = time.monotonic()
    workers = launch_shards(args, specs, len(task_configs))
    merge_results(args.output_dir, args.num_shards, len(task_configs), args.batch_size, workers)
    profile = collect_task_timings(args.output_dir)
    (args.output_dir / "task_timings.json").write_text(json.dumps(profile, indent=2) + "\n")
    report = {
        "wall_seconds": round(time.monotonic() - started, 3),
        "workers": workers,
        "estimated_worker_seconds": manifest["estimated_worker_seconds"],
        "timing_coverage": manifest["timing_coverage"],
        "gpu_samples": str(args.output_dir / "gpu_samples.jsonl"),
        "note": "Historical estimates are not a measurement of this model's runtime",
    }
    (args.output_dir / "execution_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Evaluation complete: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
