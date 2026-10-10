"""Shared/private UG, two additive branches, Down activations/ranks and widths."""

from olmo_mose.feed_forward import MoSESwiGLU
from ..registry import children, register
from .channel_control import control_spec


@register(MoSESwiGLU)
def mose(module):
    return {
        "kind": "ffn_mose",
        "children": children(module),
        "gate_nonlinearity": str(module.gate_nonlinearity),
        "up_nonlinearity": str(module.up_nonlinearity),
        "down_nonlinearity": str(module.down_nonlinearity),
        "channel_control": control_spec(module),
        "bias_shapes": {
            name: None if getattr(module, name) is None else list(getattr(module, name).shape)
            for name in ("gate_bias", "up_bias", "down_bias")
        },
    }
