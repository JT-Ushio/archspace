"""G1: export the learned alpha values, not their initial endpoints."""

from olmo_mose.adaptive_qk_attention import AdaptiveQKAttention, AdaptiveNonlinearityProjection
from ..registry import children, register
from .low_rank_attention import shared_attention


@register(AdaptiveNonlinearityProjection)
def adaptive(module):
    return {
        "kind": "adaptive",
        "alpha_shape": list(module.alpha.shape),
        "children": children(module),
    }


register(AdaptiveQKAttention)(shared_attention)
