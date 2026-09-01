from olmo_core.data import TokenizerConfig
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.transformer import TransformerConfig

from olmo_ahn import (
    LinearAttentionType,
    patch_all_linear_attention,
    patch_swa_with_linear_attention,
)


def build_olmo3_1b_baseline(tokenizer: TokenizerConfig) -> TransformerConfig:
    """Build the unmodified OLMo3-1B baseline."""
    return TransformerConfig.olmo3_1B(
        vocab_size=tokenizer.padded_vocab_size(),
        attn_backend=AttentionBackendName.flash_3,
    )


def build_all_linear_olmo3_1b(
    tokenizer: TokenizerConfig,
    attention_type: LinearAttentionType = LinearAttentionType.gdn,
) -> TransformerConfig:
    """Replace all 16 OLMo3-1B sequence mixers with linear attention."""
    return patch_all_linear_attention(
        build_olmo3_1b_baseline(tokenizer),
        attention_type=attention_type,
    )


def build_hybrid_linear_olmo3_1b(
    tokenizer: TokenizerConfig,
    attention_type: LinearAttentionType = LinearAttentionType.gdn,
) -> TransformerConfig:
    """Use linear attention on each three-layer SWA group and retain full attention."""
    return patch_swa_with_linear_attention(
        build_olmo3_1b_baseline(tokenizer),
        attention_type=attention_type,
    )
