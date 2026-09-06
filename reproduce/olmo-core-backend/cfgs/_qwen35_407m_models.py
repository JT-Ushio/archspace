"""Qwen3.5-style 407M backbone variants with the Dolma2 tokenizer."""

from typing import Optional, cast

from olmo_core.config import DType
from olmo_core.data import TokenizerConfig
from olmo_core.nn.attention import (
    AttentionBackendName,
    AttentionConfig,
    GateConfig,
    GateGranularity,
    SlidingWindowAttentionConfig,
)
from olmo_core.nn.feed_forward import FeedForwardConfig
from olmo_core.nn.layer_norm import LayerNormType
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.nn.transformer.init import InitMethod

from olmo_ahn import patch_all_linear_attention, patch_swa_with_linear_attention

D_MODEL = 1_024
N_LAYERS = 16
FFN_HIDDEN_SIZE = 3_584

FULL_N_HEADS = 8
FULL_N_KV_HEADS = 1
FULL_HEAD_DIM = 256

GDN_N_HEADS = 8
GDN_N_V_HEADS = 16
GDN_HEAD_DIM = 128
GDN_VALUE_HEAD_DIM = 128
GDN_CONV_SIZE = 4

ROPE_THETA = 10_000_000
PARTIAL_ROTARY_FACTOR = 0.25
SWA_WINDOW_SIZE = 4_096
NORM_EPS = 1e-6

TARGET_HYBRID_TOTAL_PARAMS = 510_235_520
TARGET_HYBRID_SCALING_PARAMS = 407_475_072


def _build_full_attention_backbone(
    tokenizer: TokenizerConfig,
    *,
    sliding_window: Optional[SlidingWindowAttentionConfig] = None,
) -> TransformerConfig:
    config = TransformerConfig.llama_like(
        d_model=D_MODEL,
        vocab_size=tokenizer.padded_vocab_size(),
        n_layers=N_LAYERS,
        n_heads=FULL_N_HEADS,
        n_kv_heads=FULL_N_KV_HEADS,
        head_dim=FULL_HEAD_DIM,
        gate=GateConfig(granularity=GateGranularity.elementwise),
        qk_norm=True,
        use_head_qk_norm=True,
        layer_norm_eps=NORM_EPS,
        layer_norm_name=LayerNormType.qwen_rms,
        rope_theta=ROPE_THETA,
        rope_full_precision=True,
        feed_forward=FeedForwardConfig(
            hidden_size=FFN_HIDDEN_SIZE,
            bias=False,
            dtype=DType.float32,
        ),
        attn_backend=AttentionBackendName.flash_3,
        sliding_window=sliding_window,
        dtype=DType.float32,
        tie_word_embeddings=False,
        init_method=InitMethod.scaled_truncated_normal,
    )
    attention = cast(AttentionConfig, config.block.sequence_mixer)
    attention.dropout = 0.0
    assert attention.rope is not None
    attention.rope.partial_rotary_factor = PARTIAL_ROTARY_FACTOR
    return config


def _swa_pattern(*pattern: int) -> SlidingWindowAttentionConfig:
    return SlidingWindowAttentionConfig(
        pattern=list(pattern),
        force_full_attention_on_first_layer=False,
        force_full_attention_on_last_layer=False,
    )


def _qwen35_gdn_kwargs() -> dict:
    return {
        "n_heads": GDN_N_HEADS,
        "n_v_heads": GDN_N_V_HEADS,
        "head_dim": GDN_HEAD_DIM,
        "expand_v": GDN_VALUE_HEAD_DIM / GDN_HEAD_DIM,
        "conv_size": GDN_CONV_SIZE,
        "allow_neg_eigval": False,
        "norm_eps": NORM_EPS,
    }


def build_qwen35_407m_all_gdn(tokenizer: TokenizerConfig) -> TransformerConfig:
    """Replace all 16 sequence mixers with Qwen3.5-shaped GDN."""
    return patch_all_linear_attention(
        _build_full_attention_backbone(tokenizer),
        **_qwen35_gdn_kwargs(),
    )


def build_qwen35_407m_gdn3_full1(tokenizer: TokenizerConfig) -> TransformerConfig:
    """Use the Qwen3.5 3 x GDN + 1 x full-attention pattern."""
    full_pattern = _build_full_attention_backbone(
        tokenizer,
        sliding_window=_swa_pattern(
            SWA_WINDOW_SIZE,
            SWA_WINDOW_SIZE,
            SWA_WINDOW_SIZE,
            -1,
        ),
    )
    return patch_swa_with_linear_attention(full_pattern, **_qwen35_gdn_kwargs())


def build_qwen35_407m_all_full(tokenizer: TokenizerConfig) -> TransformerConfig:
    """Use full attention in all 16 layers."""
    return _build_full_attention_backbone(tokenizer)


def build_qwen35_407m_swa3_full1(tokenizer: TokenizerConfig) -> TransformerConfig:
    """Use three 4K SWA layers followed by one full-attention layer."""
    return _build_full_attention_backbone(
        tokenizer,
        sliding_window=_swa_pattern(
            SWA_WINDOW_SIZE,
            SWA_WINDOW_SIZE,
            SWA_WINDOW_SIZE,
            -1,
        ),
    )


def build_qwen35_407m_all_swa(tokenizer: TokenizerConfig) -> TransformerConfig:
    """Use 4K sliding-window attention in all 16 layers."""
    return _build_full_attention_backbone(
        tokenizer,
        sliding_window=_swa_pattern(SWA_WINDOW_SIZE),
    )
