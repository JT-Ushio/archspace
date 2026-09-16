"""Read-only paired checkpoint analysis. Each worker owns disjoint sequences.

No distributed model wrapping, optimizer state, training or W&B writes. GPU
workers load trusted DCP model tensors independently; rank0 prepares fixed data.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
BACKEND = HERE.parents[2]
sys.path[:0] = [str(BACKEND / "src"), str(HERE)]
import olmo_mose  # register checkpoint classes
from checkpoint_io import load_model, pick_indices, resolve_checkpoint, sha256, write_csv, write_json
from metrics import attention_statistics
from olmo_core.data import NumpyPaddedFSLDatasetConfig
from olmo_core.data.utils import get_labels
from olmo_core.float8 import Float8Config

LOG = logging.getLogger("qk_analysis")
VARIANTS = {"L": "", "Q": "q", "K": "k"}


def audit_configs(configs):
    reference = None
    for (label, step), raw in configs.items():
        model = copy.deepcopy(raw["model"])
        ff, att = model["block"]["feed_forward"], model["block"]["sequence_mixer"]
        if model.get("block_overrides") or model.get("block_pattern") or model["block"]["name"] != "reordered_norm":
            raise ValueError("Expected uniform reordered_norm architecture")
        for projection in "qkvo":
            expected = "silu" if projection in VARIANTS[label] else "identity"
            if att[f"{projection}_nonlinearity"] != expected or att.get(f"{projection}_rank") not in (None, 512):
                raise ValueError(f"{label}/{step}: unexpected {projection} activation/rank")
            att[f"{projection}_nonlinearity"] = "comparison"
        if att["rank"] != 512 or att.get("share_scope", "none") != "none":
            raise ValueError("Expected unshared rank512 attention")
        if (ff["r1"], ff["r2"], ff["down_r1"], ff["down_r2"]) != (0, 512, 0, 512):
            raise ValueError("Expected rank512 single-branch FFN")
        if any(ff[f"{p}_nonlinearity"] != "identity" for p in ("gate", "up", "down")):
            raise ValueError("All FFN low-rank intermediates must remain Identity")
        if ff.get("share_gate_up_subspace") or ff["control"] != "standard" or ff["control_scope"] != "none":
            raise ValueError("Expected independent Gate/Up and unchanged primary SwiGLU")
        tm, trainer = raw["train_module"], raw["trainer"]
        if not tm["float8_config"]["enabled"]:
            raise ValueError("Expected FP8-trained checkpoints")
        if trainer["max_duration"]["value"] != 30_000_000_000 or trainer["max_duration"]["unit"] != "tokens":
            raise ValueError("Expected full30B training horizon")
        if raw["data_loader"]["global_batch_size"] != 256 * 8192:
            raise ValueError("Expected global batch256 x8192")
        if tm["optim"]["lr"] != .005 or tm["scheduler"]["warmup"] != 2000 or tm["scheduler"]["decay_fraction"] != .2:
            raise ValueError("Unexpected optimizer/schedule")
        comparable = {"model": model, "train_module": tm, "dataset": raw["dataset"],
                      "data_loader": raw["data_loader"], "init_seed": raw.get("init_seed"),
                      "eval_dataset": trainer["callbacks"]["lm_evaluator"]["eval_dataset"]}
        if reference is not None and comparable != reference:
            raise ValueError(f"{label}/{step}: training/model/data recipes differ beyond selected Q/K activations")
        reference = comparable


def prepare_samples(raw, args):
    data = copy.deepcopy(raw["trainer"]["callbacks"]["lm_evaluator"]["eval_dataset"])
    if data["sequence_length"] != 8192:
        raise ValueError("Expected 8K validation sequences")
    data["work_dir"] = str(args.work_dir)
    if args.data_root:
        if data.get("paths"):
            raise ValueError("Cannot remap an explicitly pathed evaluation dataset")
        data["mix_base_dir"] = args.data_root
    cfg = NumpyPaddedFSLDatasetConfig.from_dict(data)
    dataset = cfg.build()
    dataset.prepare()
    selected = pick_indices(dataset.offsets, args.num_sequences, args.seed)
    samples, metadata = [], []
    for sid, index in enumerate(selected):
        item = dataset[index]
        if not bool(item.get("instance_mask", True)):
            raise ValueError(f"Filtered validation sample {index}")
        ids = item["input_ids"].long()
        valid = ids.ne(cfg.tokenizer.pad_token_id)
        length = int(valid.sum())
        if length < 2 or not valid[:length].all() or valid[length:].any():
            raise ValueError("Analysis requires a right-padded single-document valid prefix")
        batch = {"input_ids": ids[None, :], "attention_mask": valid[None, :]}
        if "label_mask" in item:
            batch["label_mask"] = item["label_mask"][None, :]
        labels = get_labels(batch)[0]
        eligible = (labels != -100).nonzero().flatten()
        if not len(eligible):
            raise ValueError(f"No scored tokens in sample {index}")
        generator = torch.Generator().manual_seed(args.seed + index)
        positions = eligible[torch.randperm(len(eligible), generator=generator)[:args.tokens_per_sequence]].sort().values
        query_positions = positions[torch.linspace(0, len(positions)-1, min(args.queries, len(positions))).long()]
        source = next(i for i, (start, end) in enumerate(dataset.offsets) if start <= index < end)
        meta = {"sample": sid, "dataset_index": index, "source": str(dataset.paths[source]),
                "input_sha256": hashlib.sha256(ids.numpy().tobytes()).hexdigest(),
                "valid_length": length, "eligible_labels": len(eligible),
                "positions": positions.tolist(), "query_positions": query_positions.tolist()}
        metadata.append(meta)
        samples.append({"ids": ids, "labels": labels[positions], "positions": positions,
                        "queries": query_positions, "meta": meta})
    return samples, metadata, data


def wait_for(path, output, timeout):
    deadline = time.monotonic() + timeout
    while not path.exists():
        errors = list(output.glob("rank-*.error.json"))
        if errors:
            raise RuntimeError(f"A worker failed; inspect {errors}")
        if time.monotonic() > deadline:
            raise TimeoutError(f"Timed out waiting for {path}")
        time.sleep(2)


def run_model(model, samples, label, step, precision, output):
    rows, token_rows, handles = [], [], []
    context = {}
    for layer, block in enumerate(model.blocks.values()):
        backend = block.attention.backend

        def hook(module, inputs, kwargs, layer=layer):
            if any(kwargs.get(k) is not None for k in ("cu_doc_lens", "cu_doc_lens_q", "cu_doc_lens_k", "kv_cache_manager", "local_k_slice")):
                raise ValueError("Packed/cache/context-parallel attention is unsupported by these statistics")
            q, k, _ = inputs[0]
            sample = context["sample"]
            metrics, bins = attention_statistics(q, k, sample["queries"].to(q.device),
                                                sample["meta"]["valid_length"], module.window_size, module.scale)
            for head in range(q.shape[2]):
                rows.append({"variant": label, "step": step, "precision": precision,
                             "sample": sample["meta"]["sample"], "layer": layer, "head": head,
                             "window_left": module.window_size[0], "queries": len(sample["queries"]),
                             **{key: value[head] for key, value in metrics.items()}})
            if sample["meta"]["sample"] < 2 and layer in (0, 3, 7, 11, 15) and step == 14306 and precision == "bf16":
                import numpy as np
                np.savez_compressed(output / f"attention-{label}-{step}-sample{sample['meta']['sample']}-layer{layer}.npz",
                                    mass=bins.numpy(), query_positions=sample["queries"].numpy(),
                                    valid_length=sample["meta"]["valid_length"], key_bin_width=128)

        handles.append(backend.register_forward_pre_hook(hook, with_kwargs=True))
    try:
        with torch.no_grad():
            for sample in samples:
                context["sample"] = sample
                positions = sample["positions"].cuda()
                logits = model(sample["ids"][None, :].cuda(), logits_to_keep=positions[None, :])
                if not isinstance(logits, torch.Tensor) or logits.shape[1] != len(positions):
                    raise ValueError("Unexpected sampled logits output")
                ce = F.cross_entropy(logits[0].float(), sample["labels"].cuda(), reduction="none")
                if not torch.isfinite(ce).all():
                    raise ValueError("Nonfinite token CE")
                for position, target, loss in zip(sample["positions"].tolist(), sample["labels"].tolist(), ce.cpu().tolist()):
                    token_rows.append({"variant": label, "step": step, "precision": precision,
                                       "sample": sample["meta"]["sample"], "position": position,
                                       "target_id": target, "ce": loss,
                                       "eligible_labels": sample["meta"]["eligible_labels"],
                                       "sampled_labels": len(positions), "source": sample["meta"]["source"]})
                LOG.info("%s step%s %s sample%s CE=%.6f", label, step, precision,
                         sample["meta"]["sample"], ce.mean().item())
    finally:
        for handle in handles:
            handle.remove()
    return rows, token_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="append", required=True, help="L|Q|K=stage1-directory")
    parser.add_argument("--steps", type=int, nargs="+", default=[3576, 7152, 10728, 14306])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--num-sequences", type=int, default=128)
    parser.add_argument("--tokens-per-sequence", type=int, default=128)
    parser.add_argument("--queries", type=int, default=32)
    parser.add_argument("--seed", type=int, default=34521)
    parser.add_argument("--wait-seconds", type=int, default=3600)
    args = parser.parse_args()
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    logging.basicConfig(level=logging.INFO, format=f"%(asctime)s rank{rank} %(message)s")
    torch.set_num_threads(2)
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.num_sequences < world or min(args.tokens_per_sequence, args.queries) < 1:
        raise ValueError("Need positive sample counts and at least one sequence per GPU")
    roots = dict(value.split("=", 1) for value in args.checkpoint)
    if set(roots) != set(VARIANTS):
        raise ValueError("Exactly L/Q/K checkpoint roots are required")
    output = args.output_dir
    if rank == 0:
        output.mkdir(parents=True, exist_ok=False)
    else:
        wait_for(output, output.parent, args.wait_seconds)
    try:
        configs, checkpoints = {}, {}
        for label, root in roots.items():
            for step in args.steps:
                checkpoint, actual_step = resolve_checkpoint(Path(root) / f"step{step}")
                if actual_step != step:
                    raise ValueError("Checkpoint step mismatch")
                checkpoints[label, step] = checkpoint
                configs[label, step] = json.loads((checkpoint / "config.json").read_text())
        audit_configs(configs)
        ready = output / "samples.ready"
        if rank == 0:
            manifest = {"status": "running", "world_size": world, "arguments": vars(args).copy(),
                        "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
                        "gpu": torch.cuda.get_device_name(), "analysis_sha256": sha256(__file__),
                        "metrics_sha256": sha256(HERE / "metrics.py"),
                        "support_code_sha256": {name: sha256(HERE / name) for name in ("checkpoint_io.py", "report.py", "plots.py")},
                        "inference_batch_sequences": 1,
                        "inference_context_length": 8192,
                        "checkpoints": [{"variant": k[0], "step": k[1], "path": str(p),
                                         "config_sha256": sha256(p / "config.json"),
                                         "metadata_sha256": sha256(p / "model_and_optim/.metadata")}
                                        for k, p in checkpoints.items()]}
            manifest["arguments"] = {k: str(v) if isinstance(v, Path) else v for k, v in manifest["arguments"].items()}
            write_json(output / "manifest.json", manifest)
            samples, metadata, data = prepare_samples(next(iter(configs.values())), args)
            torch.save(samples, output / "samples.pt")
            write_json(output / "samples.json", metadata)
            write_json(output / "eval_dataset.json", data)
            ready.touch()
        else:
            wait_for(ready, output, args.wait_seconds)
            samples = torch.load(output / "samples.pt", weights_only=True)
        local_samples = samples[rank::world]
        worker = output / f"rank-{rank:02d}"
        worker.mkdir(exist_ok=False)
        for (label, step), checkpoint in checkpoints.items():
            raw = configs[label, step]
            LOG.info("Loading %s step%s", label, step)
            model = load_model(raw, checkpoint, torch.device("cuda"))
            for precision in ("bf16", "fp8"):
                if precision == "fp8":
                    ignored = set(raw["train_module"]["float8_config"].get("modules_to_ignore") or []) | {"lm_head.w_out"}
                    expected = {name for name, module in model.named_modules()
                                if isinstance(module, torch.nn.Linear) and name not in ignored
                                and all(d % 16 == 0 for d in module.weight.shape)}
                    model.apply_fp8(Float8Config.from_dict(raw["train_module"]["float8_config"]))
                    model.eval().requires_grad_(False)
                    converted = {name: {"module": str(module), "config": repr(module.config)} for name, module in model.named_modules()
                                 if type(module).__name__ == "Float8Linear"}
                    if not converted or set(converted) != expected:
                        raise ValueError("Unexpected Float8 conversion coverage")
                    write_json(worker / f"{label}-{step}-float8-modules.json", converted)
                rows, tokens = run_model(model, local_samples, label, step, precision, worker)
                write_csv(worker / f"{label}-{step}-{precision}-attention.csv", rows)
                write_csv(worker / f"{label}-{step}-{precision}-tokens.csv", tokens)
            del model
            gc.collect()
            torch.cuda.empty_cache()
        write_json(output / f"rank-{rank:02d}.complete.json", {"samples": [x["meta"]["sample"] for x in local_samples]})
        if rank == 0:
            for r in range(world):
                wait_for(output / f"rank-{r:02d}.complete.json", output, args.wait_seconds)
            from report import build_report
            build_report(output)
            manifest["status"] = "complete"
            write_json(output / "manifest.json", manifest)
            LOG.info("ALL ANALYSIS WORKERS COMPLETE: %s", output)
    except Exception as error:
        write_json(output / f"rank-{rank:02d}.error.json", {"error": str(error), "type": type(error).__name__})
        raise


if __name__ == "__main__":
    main()
