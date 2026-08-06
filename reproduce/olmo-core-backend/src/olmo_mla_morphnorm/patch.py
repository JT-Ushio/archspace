from typing import Optional

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.attention import AttentionBackendName, AttentionConfig, AttentionType
from olmo_core.nn.transformer import TransformerBlockConfig, TransformerConfig

from .config import MLAAttentionConfig
from .max_logits import MaxLogitsAttentionConfig
from .norm_types import MLANormType


def _convert_attention_config(
    attention: AttentionConfig,
    *,
    q_lora_rank: Optional[int],
    kv_lora_rank: int,
    rope_dim: int,
    value_head_dim: Optional[int],
    norm_type: MLANormType,
    use_q_a_layernorm: bool,
    use_kv_a_layernorm: Optional[bool],
    morphnorm_eps: float,
    morphnorm_update_stats: bool,
    return_max_logits: bool,
    log_max_logits_per_head: bool,
) -> MLAAttentionConfig:
    return MLAAttentionConfig(
        name=AttentionType.default,
        n_heads=attention.n_heads,
        n_kv_heads=attention.n_kv_heads,
        head_dim=attention.head_dim,
        bias=attention.bias,
        gate=None if attention.gate is None else attention.gate.copy(),
        rope=None if attention.rope is None else attention.rope.copy(),
        clip_qkv=attention.clip_qkv,
        qk_norm=None if attention.qk_norm is None else attention.qk_norm.copy(),
        dropout=attention.dropout,
        use_flash=attention.use_flash,
        backend=attention.backend,
        dtype=attention.dtype,
        sliding_window=(
            None if attention.sliding_window is None else attention.sliding_window.copy()
        ),
        use_head_qk_norm=True,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=kv_lora_rank,
        rope_dim=rope_dim,
        value_head_dim=value_head_dim,
        norm_type=norm_type,
        use_q_a_layernorm=use_q_a_layernorm,
        use_kv_a_layernorm=use_kv_a_layernorm,
        morphnorm_eps=morphnorm_eps,
        morphnorm_update_stats=morphnorm_update_stats,
        return_max_logits=return_max_logits,
        log_max_logits_per_head=log_max_logits_per_head,
    )


def patch_transformer_config(
    config: TransformerConfig,
    *,
    q_lora_rank: Optional[int] = 1024,
    kv_lora_rank: int = 512,
    rope_dim: int = 64,
    value_head_dim: Optional[int] = None,
    norm_type: MLANormType = MLANormType.morphnorm,
    use_q_a_layernorm: bool = True,
    use_kv_a_layernorm: Optional[bool] = None,
    morphnorm_eps: float = 1e-6,
    morphnorm_update_stats: bool = True,
    return_max_logits: bool = False,
    log_max_logits_per_head: bool = True,
) -> TransformerConfig:
    """
    Return a copy of an OLMo transformer config with attention replaced by MLA.

    Block type, feed-forward modules, residual/norm ordering, sliding-window patterns, RoPE
    parameters, and per-layer YaRN overrides are copied unchanged.
    """
    norm_type = MLANormType(norm_type)
    patched = config.copy()

    def patch_block(block: TransformerBlockConfig) -> None:
        attention = block.sequence_mixer
        if not isinstance(attention, AttentionConfig):
            raise OLMoConfigurationError(
                "patch_transformer_config requires every patched block to use AttentionConfig; "
                f"got {type(attention).__name__}"
            )
        block.sequence_mixer = _convert_attention_config(
            attention,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            rope_dim=rope_dim,
            value_head_dim=value_head_dim,
            norm_type=norm_type,
            use_q_a_layernorm=use_q_a_layernorm,
            use_kv_a_layernorm=use_kv_a_layernorm,
            morphnorm_eps=morphnorm_eps,
            morphnorm_update_stats=morphnorm_update_stats,
            return_max_logits=return_max_logits,
            log_max_logits_per_head=log_max_logits_per_head,
        )

    if isinstance(patched.block, dict):
        for block in patched.block.values():
            patch_block(block)
    else:
        patch_block(patched.block)
    if patched.block_overrides is not None:
        for block in patched.block_overrides.values():
            patch_block(block)
    return patched


def patch_attention_for_max_logits(
    config: TransformerConfig,
    *,
    log_max_logits_per_head: bool = True,
) -> TransformerConfig:
    """Return a copy using standard OLMo attention with the custom max-logit FA3 backend."""
    patched = config.copy()

    def patch_block(block: TransformerBlockConfig) -> None:
        attention = block.sequence_mixer
        if isinstance(attention, MLAAttentionConfig) or not isinstance(attention, AttentionConfig):
            raise OLMoConfigurationError(
                "patch_attention_for_max_logits requires standard OLMo AttentionConfig; "
                f"got {type(attention).__name__}"
            )
        block.sequence_mixer = MaxLogitsAttentionConfig(
            name=attention.name,
            n_heads=attention.n_heads,
            n_kv_heads=attention.n_kv_heads,
            head_dim=attention.head_dim,
            bias=attention.bias,
            gate=None if attention.gate is None else attention.gate.copy(),
            rope=None if attention.rope is None else attention.rope.copy(),
            clip_qkv=attention.clip_qkv,
            qk_norm=None if attention.qk_norm is None else attention.qk_norm.copy(),
            dropout=attention.dropout,
            use_flash=None,
            backend=AttentionBackendName.flash_3,
            dtype=attention.dtype,
            sliding_window=(
                None if attention.sliding_window is None else attention.sliding_window.copy()
            ),
            use_head_qk_norm=attention.use_head_qk_norm,
            log_max_logits_per_head=log_max_logits_per_head,
        )

    if isinstance(patched.block, dict):
        for block in patched.block.values():
            patch_block(block)
    else:
        patch_block(patched.block)
    if patched.block_overrides is not None:
        for block in patched.block_overrides.values():
            patch_block(block)
    return patched
