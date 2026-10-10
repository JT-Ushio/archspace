"""Bounded token-level snapshots. NumPy-only file format; no pickle or raw text.

Feature rows use fixed, increasing sampled token positions, NOT contiguous text.
Metrics and row RMS use ALL channels, even where stored feature channels are sliced.
"""
from pathlib import Path
import json

import numpy as np

POINTS = ("h", "a", "z", "y", "o")


class FeatureRecorder:
    def __init__(self, directory, sequences=2, tokens=64, channels=128):
        if min(sequences, tokens, channels) <= 0:
            raise ValueError("Snapshot limits must be positive")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.sequences, self.tokens, self.channels = sequences, tokens, channels
        self.active = False

    def begin(self, order, positions, token_ids, record):
        self.active = order < self.sequences
        if not self.active:
            return
        self.order = order
        self.rows = np.linspace(0, len(positions) - 1, min(self.tokens, len(positions)), dtype=int)
        selected = np.asarray(positions)[self.rows]
        self.arrays = {"positions": selected.astype(np.int64),
                       "token_ids": np.asarray(token_ids)[selected].astype(np.int64)}
        self.record = dict(record, sample_order=order, heatmap_positions=selected.tolist(),
                           precision="BF16 natural forward exported as float32",
                           channel_selection="a/z: all; h/y/o: evenly spaced fixed indices",
                           metrics="Full-channel RMS, negative coordinate fraction, negative energy share",
                           caveat="Same tokens across models. Channel semantics are not aligned across independently trained models.")

    def add(self, layer, point, tensor):
        if not self.active:
            return
        # Transfer only the bounded snapshot rows from the existing sampled tensor.
        if isinstance(tensor, np.ndarray):
            x = tensor[self.rows].astype(np.float32)
        else:
            x = tensor[self.rows.tolist()].detach().float().cpu().numpy()
        if point not in POINTS or x.ndim != 2 or not np.isfinite(x).all():
            raise ValueError("Invalid/nonfinite token snapshot")
        key = f"L{layer:02d}_{point}"
        if key + "_values" in self.arrays:
            raise ValueError(f"Duplicate snapshot hook: {key}")
        channels = np.arange(x.shape[1]) if point in ("a", "z") else np.linspace(
            0, x.shape[1] - 1, min(self.channels, x.shape[1]), dtype=int)
        # Float64 avoids overflowing squared raw features during statistics.
        energy = np.square(x.astype(np.float64))
        total = energy.sum(axis=1)
        neg_energy = np.divide((energy * (x < 0)).sum(axis=1), total,
                               out=np.zeros_like(total), where=total > 0)
        self.arrays[key + "_values"] = x[:, channels]
        self.arrays[key + "_channels"] = channels
        self.arrays[key + "_metrics"] = np.stack(
            [np.sqrt(energy.mean(axis=1)), (x < 0).mean(axis=1), neg_energy], axis=1)

    def finish(self, layers):
        if not self.active:
            return
        expected = {f"L{layer:02d}_{point}_values" for layer in range(1, layers + 1) for point in POINTS}
        if {k for k in self.arrays if k.endswith("_values")} != expected:
            raise ValueError("Missing or unexpected layer/point snapshots")
        stem = self.directory / f"sequence_{self.order:03d}"
        if stem.with_suffix(".npz").exists():
            raise FileExistsError(stem)
        np.savez_compressed(stem.with_suffix(".npz"), **self.arrays)
        stem.with_suffix(".json").write_text(json.dumps(self.record, indent=2, ensure_ascii=False) + "\n")
        self.arrays = {}
        self.active = False
