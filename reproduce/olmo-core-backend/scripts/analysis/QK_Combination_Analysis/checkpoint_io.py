"""Trusted-checkpoint I/O adapted from the existing Down analysis.

Model tensors only; never load optimizer state. Reject incomplete checkpoints.
"""
import copy
import csv
import hashlib
import json
from pathlib import Path
import random
import re

import torch
import olmo_mose  # register serialized model classes
from olmo_core.distributed.checkpoint import get_checkpoint_metadata, load_model_and_optim_state
from olmo_core.nn.transformer import TransformerConfig


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
