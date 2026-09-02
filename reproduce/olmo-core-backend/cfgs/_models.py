from olmo_core.data import TokenizerConfig
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.transformer import TransformerConfig

from olmo_mose import (
    LowRankAttentionSharingScope,
    MoSENonlinearity,
    SwiGLUChannelControl,
    SwiGLUChannelControlScope,
    patch_low_rank_attention,
    patch_mose_swiglu,
    patch_swiglu_channel_control,
)


def build_olmo3_1b(
    tokenizer: TokenizerConfig,
    control: SwiGLUChannelControl = SwiGLUChannelControl.standard,
    control_scope: SwiGLUChannelControlScope = SwiGLUChannelControlScope.both,
) -> TransformerConfig:
    """Build the official OLMo3-1B architecture with the selected SwiGLU control."""
    config = TransformerConfig.olmo3_1B(
        vocab_size=tokenizer.padded_vocab_size(),
        attn_backend=AttentionBackendName.flash_3,
    )
    return patch_swiglu_channel_control(
        config,
        control=control,
        control_scope=control_scope,
    )


def build_mose_olmo3_1b(
    tokenizer: TokenizerConfig,
    control: SwiGLUChannelControl,
    *,
    control_scope: SwiGLUChannelControlScope = SwiGLUChannelControlScope.both,
    r1: int = 880,
    r2: int = 880,
    down_r1: int = 880,
    down_r2: int = 880,
    gate_nonlinearity: MoSENonlinearity = MoSENonlinearity.silu,
    up_nonlinearity: MoSENonlinearity = MoSENonlinearity.silu,
    down_nonlinearity: MoSENonlinearity = MoSENonlinearity.silu,
    rms_norm_learnable_weight: bool = False,
    share_gate_up_subspace: bool = True,
    attention_low_rank_enabled: bool = False,
    attention_rank: int = 512,
    attention_q_nonlinearity: MoSENonlinearity = MoSENonlinearity.silu,
    attention_k_nonlinearity: MoSENonlinearity = MoSENonlinearity.silu,
    attention_v_nonlinearity: MoSENonlinearity = MoSENonlinearity.silu,
    attention_o_nonlinearity: MoSENonlinearity = MoSENonlinearity.silu,
    attention_rms_norm_learnable_weight: bool = False,
    attention_share_scope: LowRankAttentionSharingScope = LowRankAttentionSharingScope.none,
) -> TransformerConfig:
    """Build OLMo3-1B with configurable MoSE-SwiGLU projections."""
    config = TransformerConfig.olmo3_1B(
        vocab_size=tokenizer.padded_vocab_size(),
        attn_backend=AttentionBackendName.flash_3,
    )
    config = patch_low_rank_attention(
        config,
        enabled=attention_low_rank_enabled,
        rank=attention_rank,
        q_nonlinearity=attention_q_nonlinearity,
        k_nonlinearity=attention_k_nonlinearity,
        v_nonlinearity=attention_v_nonlinearity,
        o_nonlinearity=attention_o_nonlinearity,
        rms_norm_learnable_weight=attention_rms_norm_learnable_weight,
        share_scope=attention_share_scope,
    )
    return patch_mose_swiglu(
        config,
        control=control,
        control_scope=control_scope,
        r1=r1,
        r2=r2,
        down_r1=down_r1,
        down_r2=down_r2,
        gate_nonlinearity=gate_nonlinearity,
        up_nonlinearity=up_nonlinearity,
        down_nonlinearity=down_nonlinearity,
        rms_norm_learnable_weight=rms_norm_learnable_weight,
        share_gate_up_subspace=share_gate_up_subspace,
    )
