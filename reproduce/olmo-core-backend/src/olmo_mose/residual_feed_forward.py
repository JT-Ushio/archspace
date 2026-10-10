"""Parameter-free wide and Up-latent skips for independent low-rank SwiGLU.

The module returns an FFN update. OLMo's existing output norm and outer
residual remain in the block; returning a state that already contains x
would count that outer residual twice.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from olmo_core.exceptions import OLMoConfigurationError

from .feed_forward import (
    MoSENonlinearity,
    MoSESwiGLU,
    MoSESwiGLUConfig,
    SwiGLUChannelControl,
    SwiGLUChannelControlScope,
)
from .output_subspace import DownOutputRotation


def _validate_residual_ffn(settings, d_model=None):
    for name in ("wide_residual", "latent_residual"):
        if not isinstance(getattr(settings, name), bool):
            raise OLMoConfigurationError(f"{name} must be a boolean")
    if settings.r1 != 0 or settings.down_r1 != 0 or settings.r2 <= 0:
        raise OLMoConfigurationError("residual FFN requires a single r2 branch")
    if settings.r2 != settings.down_r2:
        raise OLMoConfigurationError("Up and Down latent sizes must match")
    if settings.share_gate_up_subspace or any(
        getattr(settings, name) is True
        for name in ("share_linear_gate_up_subspace", "share_nonlinear_gate_up_subspace")
    ):
        raise OLMoConfigurationError("residual FFN currently requires independent UG")
    if settings.gate_nonlinearity != MoSENonlinearity.identity or (
        settings.up_nonlinearity != MoSENonlinearity.identity
        or settings.down_nonlinearity != MoSENonlinearity.rms_norm
        or settings.rms_norm_learnable_weight
        or settings.control != SwiGLUChannelControl.standard
        or settings.control_scope != SwiGLUChannelControlScope.none
        or settings.down_output_rotation != DownOutputRotation.none
    ):
        raise OLMoConfigurationError("use UG Linear, no-gain Down RMS and native SwiGLU")
    if d_model is not None and settings.wide_residual and settings.hidden_size != d_model:
        raise OLMoConfigurationError("wide residual requires hidden_size == d_model")


@dataclass
class ResidualMoSESwiGLUConfig(MoSESwiGLUConfig):
    r1: int = 0
    r2: int = 512
    down_r1: int = 0
    down_r2: int = 512
    control: SwiGLUChannelControl = SwiGLUChannelControl.standard
    control_scope: SwiGLUChannelControlScope = SwiGLUChannelControlScope.none
    gate_nonlinearity: MoSENonlinearity = MoSENonlinearity.identity
    up_nonlinearity: MoSENonlinearity = MoSENonlinearity.identity
    down_nonlinearity: MoSENonlinearity = MoSENonlinearity.rms_norm
    share_gate_up_subspace: bool = False
    wide_residual: bool = False
    latent_residual: bool = False

    def __post_init__(self):
        super().__post_init__()
        if self.bias:
            raise OLMoConfigurationError("residual FFN requires bias-free projections")
        _validate_residual_ffn(self)

    def build(self, d_model: int, *, dtype: Optional[torch.dtype] = None,
              init_device: str = "cpu"):
        _validate_residual_ffn(self, d_model)
        kwargs = self.as_dict(exclude_none=True)
        kwargs.pop("name")
        kwargs.update(d_model=d_model, init_device=init_device)
        if self.dtype is not None:
            kwargs["dtype"] = self.dtype.as_pt()
        elif dtype is not None:
            kwargs["dtype"] = dtype
        return ResidualMoSESwiGLU(**kwargs)


class ResidualMoSESwiGLU(MoSESwiGLU):
    def __init__(self, *, wide_residual=False, latent_residual=False, **kwargs):
        self.wide_residual = wide_residual
        self.latent_residual = latent_residual
        if kwargs.get("bias", False):
            raise OLMoConfigurationError("residual FFN requires bias-free projections")
        super().__init__(**kwargs)
        _validate_residual_ffn(self, self.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_latent = self.gate_nonlinear_u(x)
        up_latent = self.up_nonlinear_u(x)
        hidden = F.silu(self.gate_nonlinear_v(gate_latent)) * self.up_nonlinear_v(up_latent)
        down_input = hidden + x if self.wide_residual else hidden
        down_latent = self.down_nonlinear_u(down_input)
        if self.latent_residual:
            down_latent = down_latent + up_latent
        out = self.down_nonlinear_v(self.down_nonlinear_norm(down_latent))
        return out + hidden if self.wide_residual else out
