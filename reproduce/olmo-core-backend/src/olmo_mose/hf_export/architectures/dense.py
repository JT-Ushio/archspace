"""Native Dense OLMo blocks, projections, norms, and resolved attention settings."""

import functools
import torch
from torch import nn
from torch.nn import functional as F
from olmo_core.nn.attention import Attention
from olmo_core.nn.feed_forward import FeedForward
from olmo_core.nn.layer_norm import LayerNorm, RMSNorm, QwenRMSNorm, FusedRMSNorm, CuTeRMSNorm
from olmo_core.nn.transformer.block import (
    TransformerBlock,
    ReorderedNormTransformerBlock,
    PeriNormTransformerBlock,
)

from ..registry import describe, register


@register(nn.Linear)
def linear(module):
    return {
        "kind": "linear",
        "in_features": module.in_features,
        "out_features": module.out_features,
        "bias": module.bias is not None,
    }


@register(nn.Embedding)
def embedding(module):
    return {
        "kind": "embedding",
        "num_embeddings": module.num_embeddings,
        "embedding_dim": module.embedding_dim,
        "padding_idx": module.padding_idx,
    }


@register(LayerNorm)
@register(RMSNorm)
@register(QwenRMSNorm)
@register(FusedRMSNorm)
@register(CuTeRMSNorm)
def norm(module):
    return {
        "kind": "norm",
        "norm_type": "layer"
        if type(module) is LayerNorm
        else ("qwen_rms" if type(module) is QwenRMSNorm else "rms"),
        "shape": list(module.normalized_shape),
        "eps": module.eps,
        "full_precision": module.full_precision,
        "weight_input_dtype": type(module) in (FusedRMSNorm, CuTeRMSNorm),
        "weight": module.weight is not None,
        "bias": module.bias is not None,
    }


def dense_ffn(module):
    fn = module.activation_fn
    activation = "silu" if fn is F.silu else None
    approximate = "none"
    if isinstance(fn, functools.partial) and fn.func is F.gelu:
        activation = "gelu"
        approximate = fn.keywords.get("approximate", "none")
    if activation is None:
        raise ValueError(f"Unsupported FFN activation: {type(module.activation_fn)}")
    return {
        "kind": "ffn_dense",
        "activation": activation,
        "gelu_approximate": approximate,
        "children": {name: describe(getattr(module, name)) for name in ("w1", "w2", "w3")},
    }


register(FeedForward)(dense_ffn)


def attention(module):
    rope = None
    if module.rope is not None:
        from olmo_core.nn.rope import RotaryEmbedding, compute_inv_freqs

        native = module.rope
        if type(native) is not RotaryEmbedding:
            raise ValueError(
                f"Add an explicit HF RoPE adapter for {type(native).__name__}; rotation layouts may differ"
            )
        if native.scaling is None:
            freq = compute_inv_freqs(native.theta, native.rotary_dim, torch.device("cpu"))
            rescale = 1.0
        else:
            freq, rescale = native.scaling.compute_scaled_inv_freq(
                theta=native.theta, dim=native.rotary_dim, device=torch.device("cpu")
            )
        rope = {
            "rotary_dim": native.rotary_dim,
            "theta": native.theta,
            "full_precision": native.full_precision,
            "inv_freq": freq.tolist(),
            "attention_rescale": float(rescale),
        }
    if module.backend.dropout_p != 0:
        raise ValueError("HF inference export currently requires zero attention dropout")
    projections = ("w_q", "w_k", "w_v", "w_out", "w_g", "q_norm", "k_norm")
    return {
        "kind": "attention",
        "n_heads": module.n_heads,
        "n_kv_heads": module.n_kv_heads,
        "head_dim": module.head_dim,
        "use_head_qk_norm": module.use_head_qk_norm,
        "clip_qkv": module.clip_qkv,
        "scale": module.backend.scale,
        "dropout": module.backend.dropout_p,
        "window_size": module.window_size,
        "rope": rope,
        "gate": None
        if module.gate is None
        else {
            "granularity": str(module.gate.granularity),
            "full_precision": module.gate.full_precision,
        },
        "children": {name: describe(getattr(module, name)) for name in projections},
        "input_aliases": [],
    }


register(Attention)(attention)


@register(TransformerBlock)
@register(ReorderedNormTransformerBlock)
@register(PeriNormTransformerBlock)
def block(module):
    norm_order = {
        TransformerBlock: "pre",
        ReorderedNormTransformerBlock: "post",
        PeriNormTransformerBlock: "peri",
    }[type(module)]
    names = ["attention", "attention_norm", "feed_forward", "feed_forward_norm"]
    if norm_order == "peri":
        names += ["post_attention_norm", "post_feed_forward_norm"]
    return {
        "kind": "block",
        "norm_order": norm_order,
        "attention_alpha": module.attention_residual_stream.alpha,
        "ffn_alpha": module.feed_forward_residual_stream.alpha,
        "attention_dropout": getattr(module.attention_residual_stream.dropout, "p", 0.0),
        "ffn_dropout": getattr(module.feed_forward_residual_stream.dropout, "p", 0.0),
        "children": {name: describe(getattr(module, name)) for name in names},
    }
