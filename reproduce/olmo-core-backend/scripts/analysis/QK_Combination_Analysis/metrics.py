"""Sampled post-RoPE attention statistics; no model mutations or full T*T matrix."""
import math
import torch


def attention_statistics(q, k, positions, valid_length, window_size=(-1, -1), scale=None):
    # q/k: [1,T,H,D], self-attention with right padding, no GQA or packed docs.
    if q.shape != k.shape or q.ndim != 4 or q.shape[0] != 1:
        raise ValueError("Expected matching single-sequence MHA Q/K tensors")
    if not 0 < valid_length <= q.shape[1]:
        raise ValueError("Invalid valid length")
    if positions.numel() == 0 or (positions < 0).any() or (positions >= valid_length).any():
        raise ValueError("Queries must be nonempty and inside the valid prefix")
    keys = torch.arange(valid_length, device=q.device)
    distance = positions[:, None] - keys[None, :]
    eligible = distance >= 0
    if window_size[0] >= 0:
        eligible &= distance <= window_size[0]
    if window_size[1] >= 0:
        eligible &= distance >= -window_size[1]
    logits = torch.einsum("qhd,khd->hqk", q[0, positions].float(), k[0, :valid_length].float())
    logits *= scale if scale is not None else q.shape[-1] ** -.5
    if not torch.isfinite(logits).all():
        raise ValueError("Nonfinite attention logits")
    count = eligible.sum(-1).float()
    zeroed = logits.masked_fill(~eligible, 0)
    mean = zeroed.sum(-1) / count
    variance = ((logits - mean[..., None]).square().masked_fill(~eligible, 0)).sum(-1) / count
    logp = logits.masked_fill(~eligible, -torch.inf).log_softmax(-1)
    prob = logp.exp()
    entropy = -(prob * logp.masked_fill(~eligible, 0)).sum(-1)
    # A single eligible key has entropy 0; normalized entropy defined here as 0.
    norm_entropy = entropy / count.log().clamp_min(1e-12)
    metrics = {
        "logit_mean": mean,
        "logit_std": variance.sqrt(),
        "logit_max": logits.masked_fill(~eligible, -torch.inf).amax(-1),
        "entropy": entropy,
        "normalized_entropy": norm_entropy,
        "max_probability": prob.amax(-1),
        "mean_distance": (prob * distance).sum(-1),
        "self_mass": (prob * (distance == 0)).sum(-1),
        "distance_1_16_mass": (prob * ((distance >= 1) & (distance <= 16))).sum(-1),
        "distance_17_128_mass": (prob * ((distance >= 17) & (distance <= 128))).sum(-1),
        "distance_129_1024_mass": (prob * ((distance >= 129) & (distance <= 1024))).sum(-1),
        "distance_over1024_mass": (prob * (distance > 1024)).sum(-1),
        "eligible_keys": count.expand(q.shape[2], -1),
    }
    # First four heads, 64 absolute key-position bins (128 positions at T=8192).
    width = math.ceil(q.shape[1] / 64)
    bins = torch.zeros((min(4, q.shape[2]), len(positions), 64), device=q.device)
    indices = (keys // width).expand(bins.shape[0], len(positions), -1)
    bins.scatter_add_(-1, indices, prob[:bins.shape[0]])
    return {name: value.mean(-1).cpu().tolist() for name, value in metrics.items()}, bins.cpu()
