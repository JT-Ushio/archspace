"""SVD initialization changes initial values, not checkpoint inference structure."""

from olmo_mose.svd_initialization import SVDInitTransformerConfig, base_config
from ..registry import register_config


@register_config(SVDInitTransformerConfig)
def svd_checkpoint_config(config):
    # Build the same Transformer graph without requiring old SVD artifacts or
    # attaching an initialization callback. All weights come from the checkpoint.
    return base_config(config)
