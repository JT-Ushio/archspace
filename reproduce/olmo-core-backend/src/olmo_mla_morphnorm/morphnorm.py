from typing import TYPE_CHECKING, Tuple

import torch
import torch.distributed as dist

from .norm_types import MLANormType

if TYPE_CHECKING:
    from .module import MLAAttention


def rms(x: torch.Tensor, eps: float) -> torch.Tensor:
    """Compute an RMS magnitude with FP32 accumulation and input-dtype output."""
    return torch.sqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + eps).to(x.dtype)


def accumulate_morphnorm_stats(
    attention: "MLAAttention", raw_k_nope: torch.Tensor, latent_rms: torch.Tensor
) -> None:
    """Accumulate detached per-KV-head MorphNorm ratios for the current full batch."""
    if (
        not attention.morphnorm_update_stats
        or not attention.training
        or attention.kv_cache_manager is not None
        or attention.nope_dim == 0
    ):
        return
    with torch.no_grad():
        k_rms = rms(raw_k_nope, attention.morphnorm_eps)
        ratio = (k_rms / latent_rms.unsqueeze(2)).float()
        attention.morphnorm_pending_sum.add_(ratio.sum(dim=(0, 1, 3)))
        attention.morphnorm_pending_count.add_(raw_k_nope.shape[0] * raw_k_nope.shape[1])


def prepare_cache_latents(
    attention: "MLAAttention", latent: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return key and value latent representations for absorbed decoding."""
    if attention.norm_type == MLANormType.standard_qk_norm:
        raise RuntimeError(
            "standard_qk_norm cannot be absorbed into an MQA latent cache; "
            "use baseline, materialize_norm, or morphnorm for cached decoding"
        )
    if attention.norm_type == MLANormType.morphnorm:
        latent_rms = rms(latent, attention.morphnorm_eps)
        return latent / latent_rms, latent
    return latent, latent


def materialize_kv(
    attention: "MLAAttention", latent: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Materialize grouped NoPE keys and values from the shared KV latent."""
    batch_size, seq_len, _ = latent.shape
    value = attention.w_v_b(latent).view(
        batch_size, seq_len, attention.n_kv_heads, attention.value_head_dim
    )
    if attention.w_k_b is None:
        raw_k_nope = latent.new_empty(batch_size, seq_len, attention.n_kv_heads, 0)
    else:
        raw_k_nope = attention.w_k_b(latent).view(
            batch_size, seq_len, attention.n_kv_heads, attention.nope_dim
        )

    key_latent, value_latent = prepare_cache_latents(attention, latent)
    if attention.norm_type == MLANormType.morphnorm and attention.nope_dim > 0:
        assert attention.gamma_k is not None
        latent_rms = rms(latent, attention.morphnorm_eps)
        scale = attention.morphnorm_scale.detach().to(raw_k_nope.dtype).clone().view(1, 1, -1, 1)
        gamma_k = attention.gamma_k.to(raw_k_nope.dtype)
        k_nope = raw_k_nope * gamma_k / latent_rms.unsqueeze(2) / scale
        accumulate_morphnorm_stats(attention, raw_k_nope, latent_rms)
    else:
        k_nope = raw_k_nope
    return k_nope, value, key_latent, value_latent


@torch.no_grad()
def commit_morphnorm_stats(attention: "MLAAttention", dry_run: bool = False) -> None:
    """Commit accumulated MorphNorm statistics after the complete backward pass."""
    if attention.norm_type != MLANormType.morphnorm or not attention.morphnorm_update_stats:
        return
    stats = torch.cat(
        (
            attention.morphnorm_pending_sum.float(),
            attention.morphnorm_pending_count.float().view(1),
        )
    )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    if not dry_run and bool(stats[-1] > 0):
        new_scale = stats[:-1] / stats[-1]
        attention.morphnorm_scale.copy_(new_scale.to(attention.morphnorm_scale.dtype))
    attention.morphnorm_pending_sum.zero_()
    attention.morphnorm_pending_count.zero_()
