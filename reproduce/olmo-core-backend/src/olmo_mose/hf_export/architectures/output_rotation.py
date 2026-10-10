"""Static, Dynamic and identity-initialized residual Down rotations."""

from olmo_mose.output_subspace import GroupedOutputRotation
from ..registry import register


@register(GroupedOutputRotation)
def rotation(module):
    return {
        "kind": "rotation",
        "mode": str(module.mode),
        "rank": module.rank,
        "d_model": module.d_model,
        "groups": module.groups,
        "a_max": module.a_max,
        "residual_gate": module.residual_gate is not None,
    }
