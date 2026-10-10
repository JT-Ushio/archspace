"""Selective FFN factorization at the Dense budget; Attention stays Dense."""

import functools
import math
from dataclasses import dataclass

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.feed_forward import (
    ActivationFunction,
    FeedForward,
    FeedForwardConfig,
    FeedForwardType,
)
from olmo_core.nn.transformer.init import InitMethod, init_linear

from .attention import NonlinearLowRankProjection
from .feed_forward import MoSENonlinearity

TARGETS = {"up": ("w3",), "gate": ("w1",), "up_gate": ("w1", "w3")}


@dataclass
class DenseFFNFactorizationConfig(FeedForwardConfig):
    target: str = "up"
    input_rank: int = 1632
    down_rank: int = 1632

    def __post_init__(self, registered_name=None):
        del registered_name
        if self.target not in TARGETS or self.name != FeedForwardType.default:
            raise OLMoConfigurationError("requires default FFN and target up, gate or up_gate")
        if self.bias is not False or self.activation != ActivationFunction.silu:
            raise OLMoConfigurationError("keep bias-free native SwiGLU")
        if any(
            isinstance(r, bool) or not isinstance(r, int) or r <= 0
            for r in (self.input_rank, self.down_rank)
        ):
            raise OLMoConfigurationError("factor ranks must be positive integers")

    def num_params(self, d_model):
        selected = len(TARGETS[self.target])
        dense_projection = d_model * self.hidden_size
        return (
            super().num_params(d_model)
            - (selected + 1) * dense_projection
            + (d_model + self.hidden_size) * (selected * self.input_rank + self.down_rank)
        )

    def build(self, d_model, *, dtype=None, init_device="cpu"):
        kwargs = self.as_dict(exclude_none=True)
        kwargs.pop("name")
        kwargs.update(d_model=d_model, init_device=init_device)
        if self.dtype is not None:
            kwargs["dtype"] = self.dtype.as_pt()
        elif dtype is not None:
            kwargs["dtype"] = dtype
        return DenseFFNFactorization(**kwargs)


class DenseFFNFactorization(FeedForward):
    """Keep outer SwiGLU; selected input factors are Linear, Down is no-gain RMS."""

    def __init__(self, *, target, input_rank, down_rank, **kwargs):
        if target not in TARGETS or kwargs.get("bias") is not False:
            raise OLMoConfigurationError("invalid target or nonzero bias")
        super().__init__(**kwargs)
        self.target, self.input_rank, self.down_rank = target, input_rank, down_rank
        for name in (*TARGETS[target], "w2"):
            dense = getattr(self, name)
            setattr(
                self,
                name,
                NonlinearLowRankProjection(
                    dense.in_features,
                    dense.out_features,
                    down_rank if name == "w2" else input_rank,
                    nonlinearity=(
                        MoSENonlinearity.rms_norm if name == "w2" else MoSENonlinearity.identity
                    ),
                    rms_norm_learnable_weight=False,
                    bias=False,
                    dtype=dense.weight.dtype,
                    init_device=str(dense.weight.device),
                ),
            )


def install_ffn_init_hook():
    current = InitMethod.init_feed_forward
    if getattr(current, "_dense_ffn_factorization_hook", False):
        return

    @functools.wraps(current)
    def initialize(self, module, *, d_model, block_idx, num_blocks, std=0.02, generator=None):
        if not isinstance(module, DenseFFNFactorization):
            return current(
                self,
                module,
                d_model=d_model,
                block_idx=block_idx,
                num_blocks=num_blocks,
                std=std,
                generator=generator,
            )
        if self != InitMethod.normal:
            raise OLMoConfigurationError("Dense-budget FFN factors require normal initialization")
        for name in ("w1", "w3", "w2"):
            projection = getattr(module, name)
            if isinstance(projection, NonlinearLowRankProjection):
                init_linear(projection.v, std=std, generator=generator)
                init_linear(projection.u, std=1 / math.sqrt(projection.rank), generator=generator)
            else:
                init_linear(projection, std=std, generator=generator)

    initialize._dense_ffn_factorization_hook = True
    InitMethod.init_feed_forward = initialize


install_ffn_init_hook()
