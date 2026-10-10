"""Exact registration for the independent UG residual FFN family."""

from olmo_mose.residual_feed_forward import ResidualMoSESwiGLU

from ..registry import register
from .mose_swiglu import mose


@register(ResidualMoSESwiGLU)
def residual_mose(module):
    spec = mose(module)
    spec.update(
        kind="ffn_residual_mose",
        wide_residual=module.wide_residual,
        latent_residual=module.latent_residual,
        latent_source="up",
        d_model=module.d_model,
        hidden_size=module.hidden_size,
    )
    return spec
