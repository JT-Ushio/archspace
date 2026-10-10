#!/usr/bin/env python3
"""Audit a provenance manifest and export through the common HF implementation."""

import argparse
import json
import shutil
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch

BACKEND = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND / "src"))

from olmo_mose.hf_export.audit import make_hf_config, structural_audit  # noqa: E402
from olmo_mose.hf_export.checkpoint import (  # noqa: E402
    CheckpointReader,
    build_native_meta,
    parse_model_config,
    write_json,
)
from olmo_mose.hf_export.exporter import export_checkpoint, plan_shards, target_dtype  # noqa: E402


def lookup(value, dotted):
    for key in dotted.split("."):
        value = value[key]
    return value


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--dtype", default="auto", choices=("auto", "bfloat16", "float32"))
    args = parser.parse_args()
    cohort = json.loads(args.manifest.read_text())
    rows = cohort["models"]
    if len({row["label"] for row in rows}) != len(rows):
        raise ValueError("Duplicate model labels")
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "cohort_status.json"
    if status_path.exists():
        raise FileExistsError(f"Use a new output root for another attempt: {status_path}")
    state = {"started_at": now(), "status": "preflight", "numerical_validation": "not_run", "entries": []}
    write_json(status_path, state)
    for row in rows:
        entry = {**row, "output": str(output / f"{row['label']}-step{row['step']}"), "status": "preflight"}
        state["entries"].append(entry)
        try:
            reader = CheckpointReader(row["checkpoint"])
            if reader.provenance()["step"] != row["step"]:
                raise ValueError("Checkpoint step differs from the manifest")
            # Compare the saved, parsed configuration with archived invariants.
            # Defaults are resolved by the native config, never by a run name.
            config_dict = parse_model_config(reader.raw).as_dict()
            for key, expected in row["expected_model"].items():
                actual = lookup(config_dict, key)
                if actual != expected:
                    raise ValueError(f"Saved config mismatch {key}: {actual!r} != {expected!r}")
            native = build_native_meta(reader.raw)
            reader.check_native(native)
            config = make_hf_config(native, reader.raw)
            audit = structural_audit(native, config)
            if audit["unique_parameters"] != row["parameters"]:
                raise ValueError(f"Parameter budget mismatch: {audit['unique_parameters']}")
            destination = Path(entry["output"])
            if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
                raise FileExistsError(destination)
            estimate = 0
            for shard in plan_shards(reader, config.architecture["parameter_aliases"], args.dtype, 10**9):
                for name, sources in shard:
                    shape, dtype = reader.tensors[sources[0]]
                    count = 1
                    for dim in shape:
                        count *= dim
                    estimate += count * torch.empty((), dtype=target_dtype(name, dtype, args.dtype)).element_size()
            entry.update(status="ready", source=reader.provenance(), structural_audit=audit, estimated_weight_bytes=estimate)
        except Exception as exc:
            entry.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            traceback.print_exc()
        write_json(status_path, state)
    required = sum(e.get("estimated_weight_bytes", 0) for e in state["entries"]) + 2 * 1024**3
    if shutil.disk_usage(output).free < required:
        state.update(status="failed", error=f"Insufficient space: need {required} bytes", finished_at=now())
        write_json(status_path, state)
        raise SystemExit(1)
    state["status"] = "converting"
    for entry in state["entries"]:
        if entry["status"] != "ready":
            continue
        entry["status"] = "converting"
        write_json(status_path, state)
        print(f"CONVERTING {entry['label']}: {entry['checkpoint']}", flush=True)
        try:
            manifest = export_checkpoint(entry["checkpoint"], entry["output"], dtype=args.dtype,
                                         max_shard_size="1GB", tokenizer=str(args.tokenizer), local_files_only=True)
            entry.update(status="exported", weight_files=manifest["weight_files"], completed_at=now())
            print(f"EXPORTED {entry['label']}: {entry['output']}", flush=True)
        except Exception as exc:
            entry.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            traceback.print_exc()
        write_json(status_path, state)
    state.update(status="complete" if all(e["status"] == "exported" for e in state["entries"]) else "failed", finished_at=now())
    write_json(status_path, state)
    raise SystemExit(0 if state["status"] == "complete" else 1)


if __name__ == "__main__":
    main()
