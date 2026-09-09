"""Read-only checkpoint analysis: BF16 natural activations, FP32 local replay.

One process / GPU, no training, optimizer loading, W&B or full-vocabulary logits.
Only trusted local OLMo checkpoints are supported (DCP metadata uses pickle).
"""
from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import itertools
import json
import logging
import math
import os
from pathlib import Path
import random
import re
import subprocess
import sys

import torch
import torch.nn.functional as F

BACKEND = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BACKEND / "src"))
import olmo_mose  # noqa: E402,F401 -- register serialized model classes
from olmo_core.distributed.checkpoint import (  # noqa: E402
    get_checkpoint_metadata, load_model_and_optim_state,
)
from olmo_core.nn.transformer import TransformerConfig  # noqa: E402

ACTIVATIONS = {"S": "silu", "I": "identity", "R": "rms_norm", "SR": "silu_rms_norm"}
POINTS = ("h", "a", "z", "y", "o")
LOG = logging.getLogger("down_analysis")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve_checkpoint(path):
    """Accept an exact step dir or a run dir; never fall back past an incomplete latest step."""
    path = Path(path).resolve()
    if not re.fullmatch(r"step\d+", path.name):
        candidates = [p for p in path.iterdir() if p.is_dir() and re.fullmatch(r"step\d+", p.name)]
        if not candidates:
            raise ValueError(f"No step directories under {path}")
        path = max(candidates, key=lambda p: int(p.name[4:]))
    for relative in ("config.json", "model_and_optim/.metadata"):
        if not (path / relative).is_file():
            raise FileNotFoundError(f"Incomplete checkpoint: {path / relative}")
    return path, int(path.name[4:])


def audit_configs(configs):
    """Fail closed on architecture / training recipe differences, except Down activation."""
    normalized = []
    for label, raw in configs.items():
        if not raw["train_module"]["float8_config"]["enabled"]:
            raise ValueError("Expected FP8-trained checkpoints (analysis itself uses BF16)")
        if raw["trainer"]["max_duration"]["value"] != 30_000_000_000 or raw["trainer"]["max_duration"]["unit"] != "tokens":
            raise ValueError("Expected the 30B training horizon, not the early 7.5B recipe")
        model = copy.deepcopy(raw["model"])
        ff = model["block"]["feed_forward"]
        att = model["block"]["sequence_mixer"]
        if ff["down_nonlinearity"] != ACTIVATIONS[label]:
            raise ValueError(f"{label}: unexpected Down activation")
        if model["block"]["name"] != "reordered_norm" or model.get("block_overrides") or model.get("block_pattern"):
            raise ValueError("Only uniform reordered_norm blocks are supported")
        if (ff["r1"], ff["r2"], ff["down_r1"], ff["down_r2"]) != (0, 512, 0, 512):
            raise ValueError("Expected the rank-512 single-branch controls")
        if ff.get("bias") or ff.get("rms_norm_learnable_weight") or ff.get("share_gate_up_subspace"):
            raise ValueError("Expected bias-free, gain-free, unshared controls")
        if ff["control"] != "standard" or ff["control_scope"] != "none":
            raise ValueError("Expected the original SwiGLU main gate")
        if any(ff[f"{x}_nonlinearity"] != "silu" for x in ("up", "gate")):
            raise ValueError("Gate/Up must retain their low-rank SiLU")
        if att["rank"] != 512 or att.get("share_scope", "none") != "none":
            raise ValueError("Expected unshared rank-512 attention")
        if any(att.get(f"{x}_rank", 512) not in (None, 512) or att[f"{x}_nonlinearity"] != "silu" for x in "qkvo"):
            raise ValueError("Expected rank-512 SiLU Q/K/V/O")
        ff["down_nonlinearity"] = "comparison"
        normalized.append({
            "model": model, "train_module": raw["train_module"],
            "max_duration": raw["trainer"]["max_duration"],
            "data_loader": raw["data_loader"], "init_seed": raw.get("init_seed"),
            "dataset": raw["dataset"],
            "eval_dataset": raw["trainer"]["callbacks"]["lm_evaluator"]["eval_dataset"],
        })
    if any(x != normalized[0] for x in normalized[1:]):
        raise ValueError("Checkpoint model/training/data configs differ beyond Down activation; inspect config.json")


def load_model(raw, checkpoint, device):
    cfg = TransformerConfig.from_dict(copy.deepcopy(raw["model"]))
    # Meta construction avoids unnecessary full random initialization. All state
    # must be filled by DCP; verify model key sets and shapes before loading.
    model = cfg.build(init_device="meta").to(dtype=torch.bfloat16).to_empty(device=device)
    metadata = get_checkpoint_metadata(checkpoint / "model_and_optim")
    saved = {k.removeprefix("model."): v for k, v in metadata.state_dict_metadata.items() if k.startswith("model.")}
    state = model.state_dict()
    if set(saved) != set(state):
        raise ValueError(f"Model keys mismatch: missing={set(state)-set(saved)}, extra={set(saved)-set(state)}")
    for name, value in state.items():
        if tuple(saved[name].size) != tuple(value.shape):
            raise ValueError(f"Checkpoint shape mismatch for {name}")
    load_model_and_optim_state(checkpoint / "model_and_optim", model, strict=True, thread_count=4)
    model.eval().requires_grad_(False)
    for name, value in model.state_dict().items():
        if not torch.isfinite(value).all():
            raise ValueError(f"Nonfinite checkpoint tensor: {name}")
    return model


def pick_indices(offsets, count, seed):
    """Nearly equal examples per source file; deterministic, no duplicates."""
    rng = random.Random(seed)
    pools = []
    for start, end in offsets:
        if end > start:
            pools.append(rng.sample(range(start, end), min(count, end - start)))
    chosen = []
    while len(chosen) < count and pools:
        for pool in pools:
            if pool and len(chosen) < count:
                chosen.append(pool.pop())
        pools = [pool for pool in pools if pool]
    if len(chosen) != count:
        raise ValueError(f"Dataset only has {len(chosen)} examples, requested {count}")
    return chosen


def get_samples(raw, args):
    from olmo_core.data import NumpyPaddedFSLDatasetConfig
    data = copy.deepcopy(raw["trainer"]["callbacks"]["lm_evaluator"]["eval_dataset"])
    if data["sequence_length"] != 8192:
        raise ValueError("Expected an 8192-token validation dataset")
    data["work_dir"] = str(args.work_dir)
    # The saved evaluation mix is authoritative; change only its mount root.
    if args.data_root:
        if data.get("paths"):
            raise ValueError("Explicit saved dataset paths: cannot safely remap using --data-root")
        data["mix_base_dir"] = args.data_root
    cfg = NumpyPaddedFSLDatasetConfig.from_dict(data)
    dataset = cfg.build()
    dataset.prepare()
    indices = pick_indices(dataset.offsets, args.num_sequences, args.seed)
    samples, manifest = [], []
    for index in indices:
        item = dataset[index]
        if not bool(item.get("instance_mask", True)):
            raise ValueError(f"Sample {index} is filtered by the dataset")
        ids = item["input_ids"].long()
        valid = ids.ne(cfg.tokenizer.pad_token_id)
        positions = valid.nonzero().flatten()
        if not len(positions):
            raise ValueError(f"Sample {index} contains only padding")
        generator = torch.Generator().manual_seed(args.seed + index)
        selected = positions[torch.randperm(len(positions), generator=generator)[:args.tokens_per_sequence]].sort().values
        source = next(i for i, (start, end) in enumerate(dataset.offsets) if start <= index < end)
        manifest.append({
            "dataset_index": index, "source": str(dataset.paths[source]),
            "source_index": source, "metadata": item.get("metadata", {}),
            "input_sha256": hashlib.sha256(ids.numpy().tobytes()).hexdigest(),
            "valid_tokens": len(positions), "sampled_positions": selected.tolist(),
        })
        samples.append((ids.unsqueeze(0), selected))
    return samples, manifest, data


def summarize(values):
    x = torch.cat(values).double()
    quantiles = torch.quantile(x, torch.tensor([.1, .5, .9, .99], dtype=torch.float64)).tolist()
    return dict(mean=x.mean().item(), p10=quantiles[0], p50=quantiles[1], p90=quantiles[2], p99=quantiles[3])


class ActivationStats:
    """Bounded sampled activations; all metrics refer to sampled non-pad tokens."""
    def __init__(self, covariance=False):
        self.metrics = {}
        self.values = []
        self.count = 0
        self.sum = self.cross = None
        self.covariance = covariance

    def update(self, x):
        x = x.detach().float()
        if not torch.isfinite(x).all():
            raise ValueError("Nonfinite activation encountered (not silently discarded)")
        energy = x.square().sum(-1)
        rms = x.square().mean(-1).sqrt()
        denominator = energy.clamp_min(1e-30)
        metrics = {
            "rms": rms, "absmax_over_rms": x.abs().amax(-1) / rms.clamp_min(1e-15),
            "negative_fraction": (x < 0).float().mean(-1),
            "negative_energy_fraction": (x.square() * (x < 0)).sum(-1) / denominator,
            "top1pct_energy_fraction": x.square().topk(max(1, math.ceil(x.shape[-1] * .01)), dim=-1).values.sum(-1) / denominator,
            "zero_rms": (rms == 0).float(),
        }
        for key, value in metrics.items():
            self.metrics.setdefault(key, []).append(value.cpu())
        # Equal deterministic coordinate samples per sequence for raw histograms.
        flat = x.flatten()
        indices = torch.linspace(0, len(flat) - 1, min(256, len(flat)), device=x.device).long()
        self.values.append(flat[indices].cpu())
        self.count += len(x)
        if self.covariance:
            # FP32 sufficient for diagnostic spectra; centered in float64 on CPU.
            if self.sum is None:
                self.sum = x.sum(0)
                self.cross = x.T @ x
            else:
                self.sum += x.sum(0)
                self.cross += x.T @ x

    def finish(self):
        result = {"sampled_tokens": self.count}
        for name, values in self.metrics.items():
            result.update({f"{name}_{k}": v for k, v in summarize(values).items()})
        values = torch.cat(self.values)
        for q in (.5, .99, .999):
            result[f"sample_abs_p{q*100:g}"] = torch.quantile(values.abs(), q).item()
        spectrum = None
        if self.covariance:
            mean = self.sum.double().cpu() / self.count
            cov = self.cross.double().cpu() / self.count - mean[:, None] * mean[None, :]
            spectrum = torch.linalg.eigvalsh((cov + cov.T) / 2).flip(0)
            result["cov_negative_eigenvalue_mass"] = (-spectrum.clamp_max(0)).sum().item()
            spectrum = spectrum.clamp_min(0)
            total = spectrum.sum()
            result["effective_rank"] = (total.square() / spectrum.square().sum().clamp_min(1e-30)).item()
            spectrum = spectrum / total.clamp_min(1e-30)
        return result, values, spectrum


def rms_reference(x, eps, weight=None):
    out = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
    return out if weight is None else out * weight.float()


def replay(a, decoder, outer_norm, inner_eps=1e-5):
    """Same recorded a and B for all variants; FP32 with TF32 disabled by caller."""
    a = a.float()
    silu = F.silu(a)
    z = {"I": a, "S": silu, "R": rms_reference(a, inner_eps), "SR": rms_reference(silu, inner_eps)}
    outputs = {}
    for label, value in z.items():
        y = F.linear(value, decoder.weight.float())
        o = rms_reference(y, outer_norm.eps, outer_norm.weight)
        outputs[label] = {"z": value, "y": y, "o": o}
    return outputs


def distances(reference, other):
    refnorm, norm = reference.norm(dim=-1), other.norm(dim=-1)
    valid = (refnorm > 1e-15) & (norm > 1e-15)
    return {
        "relative_l2": ((reference - other).norm(dim=-1) / refnorm.clamp_min(1e-15)).cpu(),
        "one_minus_cosine": (1 - F.cosine_similarity(reference[valid], other[valid], dim=-1)).clamp_min(0).cpu(),
        "undefined_cosine": (~valid).float().cpu(),
    }


class Collector:
    def __init__(self, model):
        self.stats, self.pairs, self.derivatives = {}, {}, {}
        self.handles = []
        self.positions = None
        self.calls = {}
        for key, block in model.blocks.items():
            layer = int(key) + 1
            ff = block.feed_forward
            if not ff.down_is_mose or ff.down_linear_u is not None or ff.down_bias is not None:
                raise ValueError("Local replay requires a single bias-free Down branch")
            if block.feed_forward_norm.bias is not None:
                raise ValueError("Outer RMSNorm must have no bias")
            def capture(point, layer=layer):
                def hook(module, inputs, output):
                    self.add(layer, point, output)
                return hook
            def encoder_hook(module, inputs, output, layer=layer, ff=ff, block=block):
                self.add(layer, "h", inputs[0])
                a = self.add(layer, "a", output)
                sigma = torch.sigmoid(a)
                derivative = sigma * (1 + a * (1 - sigma))
                self.derivatives.setdefault(layer, []).append((derivative.abs() < .1).float().mean(-1).cpu())
                eps = ff.down_nonlinear_norm.eps if ff.down_nonlinear_norm is not None else 1e-5
                results = replay(a, ff.down_nonlinear_v, block.feed_forward_norm, eps)
                for left, right in itertools.combinations(("I", "S", "R", "SR"), 2):
                    for point in ("z", "y", "o"):
                        key = (layer, left, right, point)
                        values = distances(results[left][point], results[right][point])
                        for name, value in values.items():
                            if len(value):
                                self.pairs.setdefault(key, {}).setdefault(name, []).append(value)
            def decoder_pre(module, inputs, layer=layer):
                self.add(layer, "z", inputs[0])
            self.handles.extend([
                ff.down_nonlinear_u.register_forward_hook(encoder_hook),
                ff.down_nonlinear_v.register_forward_pre_hook(decoder_pre),
                ff.down_nonlinear_v.register_forward_hook(capture("y")),
                block.feed_forward_norm.register_forward_hook(capture("o")),
            ])

    def add(self, layer, point, tensor):
        x = tensor.detach().reshape(-1, tensor.shape[-1]).index_select(0, self.positions).float()
        key = (layer, point)
        self.stats.setdefault(key, ActivationStats(covariance=point in ("a", "z"))).update(x)
        self.calls[key] = self.calls.get(key, 0) + 1
        return x

    def close(self):
        for handle in self.handles:
            handle.remove()

    def save(self, directory, label, expected_calls):
        rows, pair_rows, spectra, histogram, per_sequence = [], [], [], {}, []
        for (layer, point), stats in sorted(self.stats.items()):
            if self.calls[layer, point] != expected_calls:
                raise ValueError(f"Unexpected hook call count: {layer}, {point}")
            row, values, spectrum = stats.finish()
            rows.append(dict(run=label, layer=layer, point=point, **row))
            histogram[f"{layer}_{point}"] = values
            for i in range(expected_calls):
                per_sequence.append(dict(
                    run=label, layer=layer, point=point, sample_order=i,
                    sampled_tokens=len(stats.metrics["rms"][i]),
                    **{f"{name}_mean": vectors[i].mean().item() for name, vectors in stats.metrics.items()},
                ))
            if spectrum is not None:
                spectra.extend(dict(run=label, layer=layer, point=point, component=i+1, fraction=v)
                               for i, v in enumerate(spectrum.tolist()))
        for (layer, left, right, point), metrics in sorted(self.pairs.items()):
            row = dict(run=label, layer=layer, left=left, right=right, point=point)
            for name, values in metrics.items():
                row.update({f"{name}_{k}": v for k, v in summarize(values).items()})
            pair_rows.append(row)
        write_csv(directory / "activations.csv", rows)
        write_csv(directory / "replay.csv", pair_rows)
        write_csv(directory / "spectra.csv", spectra)
        write_csv(directory / "per_sequence.csv", per_sequence)
        write_csv(directory / "silu_sensitivity.csv", [dict(run=label, layer=layer, **summarize(values))
                  for layer, values in sorted(self.derivatives.items())])
        torch.save(histogram, directory / "histogram_samples.pt")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", required=True, metavar="S=/path/to/stage1")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--data-root", help="Override only validation mix mount root")
    parser.add_argument("--num-sequences", type=int, default=128)
    parser.add_argument("--tokens-per-sequence", type=int, default=128)
    parser.add_argument("--seed", type=int, default=34521)
    parser.add_argument("--expected-step", type=int, default=14306)
    parser.add_argument("--cpu-threads", type=int, default=8)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if any(int(os.environ.get(k, "1")) != 1 for k in ("WORLD_SIZE", "NODE_COUNT")):
        raise ValueError("Submit exactly one node / one process / one GPU")
    if min(args.num_sequences, args.tokens_per_sequence, args.cpu_threads) <= 0:
        parser.error("Sample counts and CPU threads must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required; do not run this on a CPU login node")
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0")
    configs, checkpoints, records = {}, {}, {}
    for spec in args.checkpoint:
        label, path = spec.split("=", 1)
        if label not in ACTIVATIONS or label in configs:
            parser.error("Each of S, I, R, SR must appear exactly once")
        checkpoint, step = resolve_checkpoint(path)
        if step != args.expected_step:
            raise ValueError(f"{label}: step {step} != expected {args.expected_step}; no silent fallback")
        checkpoints[label] = checkpoint
        configs[label] = json.loads((checkpoint / "config.json").read_text())
        records[label] = dict(checkpoint=str(checkpoint), step=step,
                              config_sha256=sha256(checkpoint / "config.json"),
                              metadata_sha256=sha256(checkpoint / "model_and_optim/.metadata"))
    if set(configs) != set(ACTIVATIONS):
        parser.error("Provide all four: S, I, R, SR")
    audit_configs(configs)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    # cached_path otherwise tries to create ~/.cache even for local reads.
    from cached_path import set_cache_dir
    set_cache_dir(args.work_dir / "down-analysis-cached-path")
    samples, sample_records, data_config = get_samples(configs["S"], args)
    write_json(args.output_dir / "samples.json", sample_records)
    write_json(args.output_dir / "evaluation_dataset.json", data_config)
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=BACKEND, text=True, capture_output=True)
    manifest = dict(
        status="running", checkpoints=records, arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        torch_version=torch.__version__, cuda_version=torch.version.cuda,
        gpu=torch.cuda.get_device_name(), git_commit=git.stdout.strip(),
        source_hashes={str(p.relative_to(BACKEND)): sha256(p) for p in
                      [Path(__file__), BACKEND / "src/olmo_mose/feed_forward.py", BACKEND / "src/olmo_mose/attention.py"]},
        natural_precision="BF16 weights/forward; original full-precision norm internals",
        replay_precision="FP32 from the SAME sampled BF16 encoder output a; TF32 disabled",
        valid_tokens=sum(s["valid_tokens"] for s in sample_records),
        sampled_tokens_per_layer=sum(len(s["sampled_positions"]) for s in sample_records),
        notes="No optimizer, no training, no loss evaluation. Padding excluded. Label masks do not exclude context tokens. Source-file-balanced examples; statistics are descriptive, not causal or multi-seed evidence.",
    )
    write_json(args.output_dir / "manifest.json", manifest)
    for label in ACTIVATIONS:
        directory = args.output_dir / label
        directory.mkdir()
        write_json(directory / "training_config.json", configs[label])
        LOG.info("%s loading %s (model only)", label, checkpoints[label])
        model = load_model(configs[label], checkpoints[label], device)
        collector = Collector(model)
        try:
            with torch.inference_mode():
                for i, (ids, positions) in enumerate(samples):
                    collector.positions = positions.to(device)
                    # Compute all transformer positions but only one vocabulary
                    # projection, avoiding the [8192,100352] logits allocation.
                    output = model(ids.to(device), logits_to_keep=1)
                    del output
                    if (i + 1) % 8 == 0 or i == 0:
                        LOG.info("%s sequence %d/%d", label, i + 1, len(samples))
            collector.save(directory, label, len(samples))
        finally:
            collector.close()
        del collector, model
        gc.collect()
        torch.cuda.empty_cache()
        LOG.info("%s complete", label)
    from plot_results import build_plots
    build_plots(args.output_dir)
    manifest["status"] = "complete"
    write_json(args.output_dir / "manifest.json", manifest)
    LOG.info("Analysis complete: %s", args.output_dir)


if __name__ == "__main__":
    main()
