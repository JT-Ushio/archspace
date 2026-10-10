"""Dense SiTU, RMS-SiTU and clipping controls with original rounding order."""

from olmo_mose.feed_forward import ChannelControlledFeedForward
from ..registry import register
from .dense import dense_ffn


def control_spec(module):
    return {
        "control": str(module.control),
        "scope": str(module.control_scope),
        "beta_gate": module.beta_gate,
        "beta_up": module.beta_up,
        "rms_beta_scale": module.rms_beta_scale,
    }


@register(ChannelControlledFeedForward)
def controlled(module):
    spec = dense_ffn(module)
    spec["channel_control"] = control_spec(module)
    return spec
