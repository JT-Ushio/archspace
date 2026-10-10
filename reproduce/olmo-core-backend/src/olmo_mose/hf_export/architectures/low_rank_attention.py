"""Independent, QK-, KV-, and QKV-shared nonlinear Attention factors."""

from olmo_mose.attention import LowRankAttention, NonlinearLowRankProjection
from ..registry import children, register
from .dense import attention


@register(NonlinearLowRankProjection)
def projection(module):
    return {
        "kind": "low_rank",
        "nonlinearity": str(module.nonlinearity),
        "children": children(module),
    }


def shared_attention(module):
    spec = attention(module)
    seen = {}
    for name in ("w_q", "w_k", "w_v"):
        projection = getattr(module, name)
        if hasattr(projection, "v"):
            if id(projection.v) in seen:
                spec["input_aliases"].append([name, seen[id(projection.v)]])
            else:
                seen[id(projection.v)] = name
    return spec


register(LowRankAttention)(shared_attention)
