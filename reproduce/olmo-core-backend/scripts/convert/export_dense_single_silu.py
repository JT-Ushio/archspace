#!/usr/bin/env python3
"""Export final Q/K/Up/Gate single-SiLU checkpoints and the original Dense control."""

import argparse
import shutil
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch

BACKEND = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND / "src"))

from olmo_mose.hf_export.audit import make_hf_config, structural_audit  # noqa: E402
from olmo_mose.hf_export.checkpoint import CheckpointReader, build_native_meta, write_json  # noqa: E402
from olmo_mose.hf_export.exporter import export_checkpoint, plan_shards, target_dtype  # noqa: E402

COHORT = [
    ("dense", "olmo3-1b-dense-baseline-bs256-seq8k-30b-wsd-fp8-5e-3", 1_484_916_736, None),
    ("q", "olmo3-1b-dense-matched-q-only-silu-r1024-fp8-30b", 1_484_916_736, "q"),
    ("k", "olmo3-1b-dense-matched-k-only-silu-r1024-fp8-30b", 1_484_916_736, "k"),
    ("up", "olmo3-1b-dense-nearmatched-up-only-silu-r1632-fp8-30b", 1_483_868_160, "up"),
    ("gate", "olmo3-1b-dense-nearmatched-gate-only-silu-r1632-fp8-30b", 1_483_868_160, "gate"),
]


def verify_cohort_model(model, target, expected_params):
    from olmo_core.nn.attention import Attention
    from olmo_core.nn.feed_forward import FeedForward
    from olmo_mose.dense_single_attention import DenseSingleSiLUAttention
    from olmo_mose.dense_single_ffn import DenseSingleSiLUFFN

    if (model.d_model, model.n_layers, sum(p.numel() for p in model.parameters())) != (
        2048,
        16,
        expected_params,
    ):
        raise ValueError("Checkpoint differs from the expected OLMo3-1B parameter budget")
    for block in model.blocks.values():
        if target in ("q", "k"):
            if (
                type(block.attention) is not DenseSingleSiLUAttention
                or block.attention.target != target
            ):
                raise ValueError(f"Expected only {target.upper()} to be factorized")
            if (
                getattr(block.attention, "w_" + target).rank != 1024
                or type(block.feed_forward) is not FeedForward
            ):
                raise ValueError("Incorrect single-attention factor rank or non-Dense FFN")
        elif target in ("up", "gate"):
            if (
                type(block.attention) is not Attention
                or type(block.feed_forward) is not DenseSingleSiLUFFN
            ):
                raise ValueError("Expected Dense Attention and a single FFN replacement")
            if block.feed_forward.target != target or block.feed_forward.rank != 1632:
                raise ValueError("Incorrect FFN factor target/rank")
        elif type(block.attention) is not Attention or type(block.feed_forward) is not FeedForward:
            raise ValueError("Dense baseline contains a custom projection")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=BACKEND / "assets/dolma2-tokenizer")
    parser.add_argument("--step", type=int, default=14306)
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float32"), default="auto")
    parser.add_argument("--max-shard-size", default="1GB")
    args = parser.parse_args()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    state = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "preflight",
        "step": args.step,
        "dtype": args.dtype,
        "numerical_validation": "not_run",
        "benchmarks": "not_selected",
        "entries": [],
    }
    status_path = output / "cohort_status.json"
    write_json(status_path, state)
    try:
        for label, name, count, target in COHORT:
            checkpoint = args.checkpoint_root / name / "stage1" / f"step{args.step}"
            reader = CheckpointReader(checkpoint)
            native = build_native_meta(reader.raw)
            reader.check_native(native)
            verify_cohort_model(native, target, count)
            config = make_hf_config(native, reader.raw)
            audit = structural_audit(native, config)
            estimate = 0
            for shard in plan_shards(
                reader, config.architecture["parameter_aliases"], args.dtype, 10**9
            ):
                for hf_name, sources in shard:
                    shape, dtype = reader.tensors[sources[0]]
                    n = 1
                    for dim in shape:
                        n *= dim
                    estimate += (
                        n
                        * torch.empty(
                            (), dtype=target_dtype(hf_name, dtype, args.dtype)
                        ).element_size()
                    )
            destination = output / f"{label}-step{args.step}"
            if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
                raise FileExistsError(f"Existing output requires a new output root: {destination}")
            state["entries"].append(
                {
                    "label": label,
                    "run_name": name,
                    "checkpoint": str(checkpoint),
                    "output": str(destination),
                    "parameters": count,
                    "exact_dense_budget": count == COHORT[0][2],
                    "source": reader.provenance(),
                    "structural_audit": audit,
                    "estimated_weight_bytes": estimate,
                    "status": "ready",
                }
            )
        required = sum(e["estimated_weight_bytes"] for e in state["entries"]) + 2 * 1024**3
        if shutil.disk_usage(output).free < required:
            raise OSError(f"Insufficient output space: need at least {required} bytes")
        write_json(
            output / "exports.json",
            [{"checkpoint": e["checkpoint"], "output": e["output"]} for e in state["entries"]],
        )
        state["status"] = "converting"
        write_json(status_path, state)
        for entry in state["entries"]:
            entry["status"] = "converting"
            write_json(status_path, state)
            print(f"CONVERTING {entry['label']}: {entry['checkpoint']}", flush=True)
            try:
                manifest = export_checkpoint(
                    entry["checkpoint"],
                    entry["output"],
                    dtype=args.dtype,
                    max_shard_size=args.max_shard_size,
                    tokenizer=str(args.tokenizer),
                    local_files_only=True,
                )
                entry["status"] = "exported"
                entry["weight_files"] = manifest["weight_files"]
                entry["completed_at"] = datetime.now(timezone.utc).isoformat()
                print(f"EXPORTED {entry['label']}: {entry['output']}", flush=True)
            except Exception as exc:
                entry["status"] = "failed"
                entry["error"] = f"{type(exc).__name__}: {exc}"
                traceback.print_exc()
            write_json(status_path, state)
        state["status"] = (
            "complete" if all(e["status"] == "exported" for e in state["entries"]) else "failed"
        )
        state["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(status_path, state)
        if state["status"] != "complete":
            raise SystemExit(1)
    except Exception as exc:
        state["status"] = "failed"
        state["error"] = f"{type(exc).__name__}: {exc}"
        write_json(status_path, state)
        raise


if __name__ == "__main__":
    main()
