from olmo_core.data import TokenizerConfig
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.transformer import TransformerConfig

from olmo_mose import (
    MoSENonlinearity,
    SwiGLUChannelControl,
    SwiGLUChannelControlScope,
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
) -> TransformerConfig:
    """Build OLMo3-1B with configurable MoSE-SwiGLU projections."""
    config = TransformerConfig.olmo3_1B(
        vocab_size=tokenizer.padded_vocab_size(),
        attn_backend=AttentionBackendName.flash_3,
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
    )
