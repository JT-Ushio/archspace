from typing import cast

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.attention import AttentionConfig
from olmo_core.nn.transformer import TransformerBlockConfig, TransformerConfig

from .linear_attention import LinearAttentionConfig, LinearAttentionType


def _require_patchable_olmo3(config: TransformerConfig) -> TransformerBlockConfig:
    if isinstance(config.block, dict) or config.block_pattern is not None:
        raise OLMoConfigurationError(
            "AHN patching requires an OLMo3-style single base block without block_pattern"
        )
    if config.block_overrides is not None:
        raise OLMoConfigurationError(
            "AHN patching requires block_overrides=None so existing per-layer changes are not lost"
        )
    if not isinstance(config.block.sequence_mixer, AttentionConfig):
        raise OLMoConfigurationError("AHN patching requires an AttentionConfig base block")
    return cast(TransformerBlockConfig, config.block)


def _linear_config(
    block: TransformerBlockConfig,
    attention_type: LinearAttentionType,
    **kwargs,
) -> LinearAttentionConfig:
    attention = cast(AttentionConfig, block.sequence_mixer)
    return LinearAttentionConfig(
        attention_type=attention_type,
        dtype=attention.dtype,
        **kwargs,
    )


def patch_all_linear_attention(
    config: TransformerConfig,
    *,
    attention_type: LinearAttentionType = LinearAttentionType.gdn,
    **kwargs,
) -> TransformerConfig:
    """Return a copy with every OLMo transformer layer replaced by linear attention."""
    base_block = _require_patchable_olmo3(config)
    patched = config.copy()
    assert isinstance(patched.block, TransformerBlockConfig)
    patched.block.sequence_mixer = _linear_config(base_block, attention_type, **kwargs)
    return patched


def patch_swa_with_linear_attention(
    config: TransformerConfig,
    *,
    attention_type: LinearAttentionType = LinearAttentionType.gdn,
    **kwargs,
) -> TransformerConfig:
    """Replace OLMo3 SWA layers while preserving every full-attention layer."""
    base_block = _require_patchable_olmo3(config)
    attention = cast(AttentionConfig, base_block.sequence_mixer)
    if attention.sliding_window is None:
        raise OLMoConfigurationError(
            "patch_swa_with_linear_attention requires an OLMo3 sliding-window pattern"
        )

    full_layer_indices = [
        layer_idx
        for layer_idx in range(config.n_layers)
        if not attention.sliding_window.should_use_swa(layer_idx, config.n_layers)
    ]
    if not full_layer_indices or len(full_layer_indices) == config.n_layers:
        raise OLMoConfigurationError(
            "the base attention pattern must contain both SWA and full-attention layers"
        )

    patched = config.copy()
    assert isinstance(patched.block, TransformerBlockConfig)
    full_block = patched.block.copy()
    patched.block.sequence_mixer = _linear_config(base_block, attention_type, **kwargs)
    patched.block_overrides = {
        layer_idx: full_block.copy() for layer_idx in full_layer_indices
    }
    return patched
