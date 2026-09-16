"""Opt-in, artifact-backed initialization; ordinary MoSE defaults are untouched.

Notation is y = D phi(E x), independently of historical U/V member names.
Artifacts are generated once in FP32, not by every FSDP rank. Normal model
initialization ALWAYS runs first, preserving its generator consumption. A later
trainer checkpoint restore naturally overwrites initial weights as usual.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import json
import logging
from pathlib import Path

import torch
from torch.distributed.tensor import DTensor
from olmo_core.distributed.utils import distribute_like, get_local_tensor
from olmo_core.nn.transformer import TransformerConfig

MODES = ("independent", "svd_raw", "svd_scale_matched")
LOG = logging.getLogger(__name__)


def tensor_hash(tensor):
    if isinstance(tensor, DTensor):
        tensor = tensor.full_tensor()
    value = tensor.detach().float().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rms(value):
    return value.float().square().mean().sqrt()


def projection_pairs(model):
    """Fail closed for unsupported sharing, bypasses, or incomplete coverage."""
    pairs = {}
    for index, block in model.blocks.items():
        att, ff = block.attention, block.feed_forward
        if str(att.share_scope) != "none" or ff.share_gate_up_subspace:
            raise ValueError("SVD initialization experiment requires unshared factors")
        if ff.r1 != 0 or ff.down_r1 != 0:
            raise ValueError("SVD initialization experiment requires no dense bypass")
        for short, attr in (("q", "w_q"), ("k", "w_k"), ("v", "w_v"), ("o", "w_out")):
            p = getattr(att, attr)
            pairs[f"blocks.{index}.{short}"] = (p.v, p.u)
        for short in ("gate", "up", "down"):
            enc, dec = getattr(ff, f"{short}_nonlinear_u"), getattr(ff, f"{short}_nonlinear_v")
            if enc is None or dec is None or enc.bias is not None or dec.bias is not None:
                raise ValueError(f"Unsupported {short} projection")
            pairs[f"blocks.{index}.{short}"] = (enc, dec)
    if len(pairs) != 7 * model.n_layers:
        raise ValueError("Incomplete seven-projection coverage")
    return pairs


def canonical_svd(weight, rank):
    """Exact reduced FP32 SVD; signs fixed by largest encoder-row entry."""
    if weight.dtype != torch.float32 or weight.ndim != 2 or not 0 < rank <= min(weight.shape):
        raise ValueError("Expected FP32 matrix and valid rank")
    # gesvd (QR) avoids randomized low-rank approximation and its extra RNG.
    kwargs = {"driver": "gesvd"} if weight.is_cuda else {}
    left, singular, right = torch.linalg.svd(weight, full_matrices=False, **kwargs)
    left, right = left[:, :rank], right[:rank]
    signs = right[torch.arange(rank, device=right.device), right.abs().argmax(dim=1)].sign()
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    root = singular[:rank].sqrt()
    enc = (root * signs)[:, None] * right
    dec = left * (root * signs)[None, :]
    return enc.contiguous(), dec.contiguous(), singular


def make_factors(enc_ref, dec_ref, *, std, seed):
    """Local generator does not advance normal model initialization RNG."""
    rank, input_dim = enc_ref.shape
    output_dim, rank_d = dec_ref.shape
    if rank != rank_d:
        raise ValueError("Mismatched pair rank")
    generator = torch.Generator(device=enc_ref.device).manual_seed(seed)
    weight = torch.empty(output_dim, input_dim, device=enc_ref.device, dtype=torch.float32)
    torch.nn.init.trunc_normal_(weight, std=std, a=-3 * std, b=3 * std, generator=generator)
    enc, dec, singular = canonical_svd(weight, rank)
    enc_scale, dec_scale = rms(enc_ref) / rms(enc), rms(dec_ref) / rms(dec)
    modes = {"svd_raw": {"E": enc, "D": dec},
             "svd_scale_matched": {"E": enc * enc_scale, "D": dec * dec_scale}}
    energy = singular.square()
    metrics = dict(seed=seed, dense_std=std, dense_shape=list(weight.shape), rank=rank,
                   dense_rms=rms(weight).item(), retained_energy_fraction=(energy[:rank].sum()/energy.sum()).item(),
                   relative_truncation_error=(energy[rank:].sum()/energy.sum()).sqrt().item(),
                   encoder_scale=enc_scale.item(), decoder_scale=dec_scale.item(),
                   product_scale=(enc_scale * dec_scale).item(),
                   baseline_E_sha256=tensor_hash(enc_ref), baseline_D_sha256=tensor_hash(dec_ref),
                   baseline_E_rms=rms(enc_ref).item(), baseline_D_rms=rms(dec_ref).item())
    return modes, metrics


def stable_seed(seed, name):
    return int.from_bytes(hashlib.sha256(f"{seed}:{name}".encode()).digest()[:8], "little") % (2**63-1)


def base_config(config):
    return TransformerConfig(**{f.name: getattr(config, f.name) for f in fields(TransformerConfig)})


def config_fingerprint(config):
    raw = base_config(config).as_config_dict()
    return hashlib.sha256(json.dumps(raw, sort_keys=True, default=str).encode()).hexdigest()


@torch.no_grad()
def apply_artifacts(model, directory, mode, fingerprint, *, require_probe=True):
    if mode not in MODES:
        raise ValueError(f"Unknown initialization: {mode}")
    root = Path(directory)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["model_sha256"] != fingerprint or manifest["status"] != "complete":
        raise ValueError("Incomplete or wrong-model SVD artifact")
    if require_probe:
        gate = json.loads((root / "probe_complete.json").read_text())
        if gate.get("status") != "passed" or gate.get("manifest_sha256") != file_hash(root / "manifest.json"):
            raise ValueError("Step0 probe not passed for this exact manifest")
    pairs = projection_pairs(model)
    if set(pairs) != set(manifest["pairs"]):
        raise ValueError("Artifact coverage mismatch")
    for name, (enc, dec) in pairs.items():
        entry = manifest["pairs"][name]
        if tensor_hash(enc.weight) != entry["baseline_E_sha256"] or tensor_hash(dec.weight) != entry["baseline_D_sha256"]:
            raise ValueError(f"Baseline initialization mismatch for {name}; refusing an unpaired experiment")
        if mode == "independent":
            continue
        path = root / entry["file"]
        if file_hash(path) != entry["sha256"]:
            raise ValueError(f"Artifact checksum mismatch: {name}")
        saved = torch.load(path, map_location="cpu", weights_only=True)[mode]
        for module, key in ((enc, "E"), (dec, "D")):
            weight = module.weight
            value = saved[key]
            if tuple(value.shape) != tuple(weight.shape) or not torch.isfinite(value).all():
                raise ValueError(f"Invalid artifact tensor: {name}/{key}")
            value = value.to(device=weight.device, dtype=weight.dtype)
            if isinstance(weight, DTensor):
                # Core API: first argument supplies the distribution layout;
                # second argument supplies the values to distribute.
                shard = distribute_like(weight, value)
                get_local_tensor(weight).copy_(get_local_tensor(shard))
            else:
                weight.copy_(value)
    LOG.info("INITIALIZATION VERIFIED mode=%s pairs=%d manifest=%s", mode, len(pairs), file_hash(root / "manifest.json"))


@dataclass
class SVDInitTransformerConfig(TransformerConfig):
    factor_init: str = "independent"
    factor_init_artifacts: str = ""

    def build(self, *, init_device="cpu"):
        if self.factor_init not in MODES or not self.factor_init_artifacts:
            raise ValueError("An explicit initialization mode and verified artifact directory are required")
        model = super().build(init_device=init_device)
        original = model.init_weights
        mode, directory, fingerprint = self.factor_init, self.factor_init_artifacts, config_fingerprint(self)

        def initialize(*args, **kwargs):
            generator = original(*args, **kwargs)
            apply_artifacts(model, directory, mode, fingerprint)
            return generator

        model.init_weights = initialize
        return model
