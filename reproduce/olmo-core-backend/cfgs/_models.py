from olmo_core.data import TokenizerConfig
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.transformer import TransformerConfig

from olmo_mla_morphnorm import (
    MLANormType,
    patch_attention_for_max_logits,
    patch_transformer_config,
)

Q_LORA_RANK = 1024
KV_LORA_RANK = 512
ROPE_DIM = 64


def build_olmo_baseline(tokenizer: TokenizerConfig) -> TransformerConfig:
    """Build the unchanged OLMo3-1B architecture with max-logit FA3 monitoring."""
    return patch_attention_for_max_logits(
        TransformerConfig.olmo3_1B(
            vocab_size=tokenizer.padded_vocab_size(),
            attn_backend=AttentionBackendName.flash_3,
        )
    )


def build_mla(tokenizer: TokenizerConfig, norm_type: MLANormType) -> TransformerConfig:
    """Build OLMo3-1B blocks with MLA and the selected normalization mode."""
    return patch_transformer_config(
        TransformerConfig.olmo3_1B(
            vocab_size=tokenizer.padded_vocab_size(),
            attn_backend=AttentionBackendName.flash_3,
        ),
        q_lora_rank=512,
        kv_lora_rank=256,
        rope_dim=64,
        head_dim=192,
        norm_type=norm_type,
        return_max_logits=True,
        log_max_logits_per_head=True,
    )
