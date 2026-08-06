from typing import TYPE_CHECKING, Optional

import torch
import torch.nn.functional as F

from .morphnorm import rms
from .norm_types import MLANormType

if TYPE_CHECKING:
    from .module import MLAAttention


def pad_last_dim(x: torch.Tensor, target_dim: int) -> torch.Tensor:
    """Right-pad a tensor's final dimension to ``target_dim``."""
    if x.shape[-1] > target_dim:
        raise ValueError(f"Cannot pad dimension {x.shape[-1]} down to {target_dim}")
    if x.shape[-1] == target_dim:
        return x
    return F.pad(x, (0, target_dim - x.shape[-1]))


def build_attention_mask(
    attention: "MLAAttention",
    *,
    batch_size: int,
    query_len: int,
    key_len: int,
    query_start: int,
    device: torch.device,
    cache_leftpad: Optional[torch.Tensor],
) -> torch.Tensor:
    """Build a causal, optional sliding-window and left-padding mask."""
    query_positions = torch.arange(query_len, device=device) + query_start
    key_positions = torch.arange(key_len, device=device)
    allowed = key_positions.view(1, 1, 1, key_len) <= query_positions.view(1, 1, query_len, 1)
    if attention.window_size is not None:
        allowed = allowed & (
            key_positions.view(1, 1, 1, key_len)
            >= (query_positions - attention.window_size + 1).view(1, 1, query_len, 1)
        )
    if cache_leftpad is not None:
        allowed = allowed & (
            key_positions.view(1, 1, 1, key_len) >= cache_leftpad.view(batch_size, 1, 1, 1)
        )
    return allowed


def materialized_attention(
    attention: "MLAAttention",
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cu_doc_lens: Optional[torch.Tensor],
    cu_doc_lens_q: Optional[torch.Tensor],
    cu_doc_lens_k: Optional[torch.Tensor],
    max_doc_len: Optional[int],
    max_doc_len_q: Optional[int],
    max_doc_len_k: Optional[int],
    local_k_slice: Optional[slice],
    cache_leftpad: Optional[torch.Tensor],
) -> torch.Tensor:
    """Run materialized MHA/GQA through the configured training/prefill backend."""
    v = pad_last_dim(v, attention.head_dim)
    if cache_leftpad is not None and not bool(cache_leftpad.any()):
        cache_leftpad = None
    if cache_leftpad is None:
        output = attention.backend(
            (q, k, v),
            cu_doc_lens=cu_doc_lens,
            cu_doc_lens_q=cu_doc_lens_q,
            cu_doc_lens_k=cu_doc_lens_k,
            max_doc_len=max_doc_len,
            max_doc_len_q=max_doc_len_q,
            max_doc_len_k=max_doc_len_k,
            local_k_slice=local_k_slice,
            kv_cache_manager=None,
        )
        return output[..., : attention.value_head_dim]

    if any(
        item is not None
        for item in (
            cu_doc_lens,
            cu_doc_lens_q,
            cu_doc_lens_k,
            max_doc_len,
            max_doc_len_q,
            max_doc_len_k,
            local_k_slice,
        )
    ):
        raise RuntimeError("Left-padded prefill cannot be combined with packed-document inputs")

    repeats = attention.n_heads // attention.n_kv_heads
    k = k.repeat_interleave(repeats, dim=2).transpose(1, 2)
    v = v.repeat_interleave(repeats, dim=2).transpose(1, 2)
    q = q.transpose(1, 2)
    mask = build_attention_mask(
        attention,
        batch_size=q.shape[0],
        query_len=q.shape[2],
        key_len=k.shape[2],
        query_start=k.shape[2] - q.shape[2],
        device=q.device,
        cache_leftpad=cache_leftpad,
    )
    output = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=mask,
        dropout_p=attention.dropout if attention.training else 0.0,
        is_causal=False,
        scale=attention.softmax_scale,
    )
    return output.transpose(1, 2)[..., : attention.value_head_dim].contiguous()


def absorbed_query(attention: "MLAAttention", q_nope: torch.Tensor) -> torch.Tensor:
    """Absorb grouped NoPE key up-projections into each query head."""
    batch_size, seq_len, _, _ = q_nope.shape
    if attention.w_k_b is None:
        return q_nope.new_zeros(batch_size, seq_len, attention.n_heads, attention.kv_lora_rank)

    group_size = attention.n_heads // attention.n_kv_heads
    weights = attention.w_k_b.weight.view(
        attention.n_kv_heads, attention.nope_dim, attention.kv_lora_rank
    )
    weights = weights.repeat_interleave(group_size, dim=0)
    q_for_absorption = q_nope
    if attention.norm_type == MLANormType.morphnorm:
        assert attention.gamma_k is not None
        q_for_absorption = q_for_absorption * attention.gamma_k.to(q_for_absorption.dtype)
    absorbed = torch.einsum("bthd,hdr->bthr", q_for_absorption, weights)
    if attention.norm_type == MLANormType.morphnorm:
        head_scale = attention.morphnorm_scale.repeat_interleave(group_size).to(absorbed.dtype)
        absorbed = absorbed / head_scale.view(1, 1, -1, 1)
    return absorbed


def decompress_values(attention: "MLAAttention", latent_output: torch.Tensor) -> torch.Tensor:
    """Apply grouped value up-projections after latent MQA attention."""
    group_size = attention.n_heads // attention.n_kv_heads
    weights = attention.w_v_b.weight.view(
        attention.n_kv_heads, attention.value_head_dim, attention.kv_lora_rank
    )
    weights = weights.repeat_interleave(group_size, dim=0)
    return torch.einsum("bthr,hvr->bthv", latent_output, weights)


def mqa_attention(
    attention: "MLAAttention",
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    latent_cache: torch.Tensor,
    rope_key: torch.Tensor,
) -> torch.Tensor:
    """Run absorbed, single-latent-head MQA decoding."""
    if attention.norm_type == MLANormType.morphnorm:
        key_latent = latent_cache / rms(latent_cache, attention.morphnorm_eps)
    else:
        key_latent = latent_cache
    q = torch.cat((absorbed_query(attention, q_nope), q_rope), dim=-1).transpose(1, 2)
    k = torch.cat((key_latent, rope_key), dim=-1).unsqueeze(1)
    v = pad_last_dim(latent_cache, k.shape[-1]).unsqueeze(1)
    batch_size, _, query_len, _ = q.shape
    key_len = k.shape[2]
    assert attention.kv_cache_manager is not None
    mask = build_attention_mask(
        attention,
        batch_size=batch_size,
        query_len=query_len,
        key_len=key_len,
        query_start=key_len - query_len,
        device=q.device,
        cache_leftpad=attention.kv_cache_manager.cache_leftpad,
    )

    if q.device.type == "cuda":
        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=0.0,
            is_causal=False,
            scale=attention.softmax_scale,
            enable_gqa=True,
        )
    else:
        k = k.expand(-1, attention.n_heads, -1, -1)
        v = v.expand(-1, attention.n_heads, -1, -1)
        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=0.0,
            is_causal=False,
            scale=attention.softmax_scale,
        )
    latent_output = output[..., : attention.kv_lora_rank].transpose(1, 2)
    return decompress_values(attention, latent_output)
