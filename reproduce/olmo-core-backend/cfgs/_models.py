from olmo_core.data import TokenizerConfig
from olmo_core.nn.transformer import TransformerConfig

from olmo_mose import (
    SwiGLUChannelControl,
    patch_mose_swiglu,
    patch_swiglu_channel_control,
)


def build_olmo3_1b(
    tokenizer: TokenizerConfig,
    control: SwiGLUChannelControl = SwiGLUChannelControl.standard,
) -> TransformerConfig:
    """Build the official OLMo3-1B architecture with the selected SwiGLU control."""
    config = TransformerConfig.olmo3_1B(vocab_size=tokenizer.padded_vocab_size())
    return patch_swiglu_channel_control(config, control=control)


def build_mose_olmo3_1b(
    tokenizer: TokenizerConfig,
    control: SwiGLUChannelControl,
    *,
    r1: int = 880,
    r2: int = 880,
    down_r1: int = 880,
    down_r2: int = 880,
) -> TransformerConfig:
    """Build OLMo3-1B with configurable MoSE-SwiGLU projections."""
    config = TransformerConfig.olmo3_1B(vocab_size=tokenizer.padded_vocab_size())
    return patch_mose_swiglu(
        config,
        control=control,
        r1=r1,
        r2=r2,
        down_r1=down_r1,
        down_r2=down_r2,
    )
