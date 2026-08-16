from typing import Optional

from olmo_core.data import TokenizerConfig
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.transformer import TransformerConfig

from olmo_cubit import patch_cubit


def build_olmo3_1b(tokenizer: TokenizerConfig) -> TransformerConfig:
    """Build the OLMo3-1B control with the same FA3 backend as Cubit."""
    return TransformerConfig.olmo3_1B(
        vocab_size=tokenizer.padded_vocab_size(),
        attn_backend=AttentionBackendName.flash_3,
    )


def build_cubit_olmo3_1b(
    tokenizer: TokenizerConfig,
    *,
    cubit_start_layer: int = 0,
    cubit_end_layer: Optional[int] = None,
    share_reference: bool = False,
    regularization: float = 1e-10,
    lrr_lower: float = 0.5,
    lrr_upper: float = 2.0,
) -> TransformerConfig:
    """Build OLMo3-1B with Cubit token mixers and an FA3 output aggregation."""
    return patch_cubit(
        build_olmo3_1b(tokenizer),
        cubit_start_layer=cubit_start_layer,
        cubit_end_layer=cubit_end_layer,
        share_reference=share_reference,
        regularization=regularization,
        lrr_lower=lrr_lower,
        lrr_upper=lrr_upper,
    )
