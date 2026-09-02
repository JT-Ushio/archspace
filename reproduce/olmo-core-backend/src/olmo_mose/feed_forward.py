import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import DeviceMesh
from torch.distributed.tensor.placement_types import Placement

from olmo_core.config import StrEnum
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.feed_forward import (
    ActivationFunction,
    FeedForward,
    FeedForwardConfig,
    FeedForwardType,
)
from olmo_core.nn.layer_norm import RMSNorm


class SwiGLUChannelControl(StrEnum):
    """Channel-control function applied inside SwiGLU."""

    standard = "standard"
    situ = "situ"
    situ_rms = "situ_rms"
    asymmetric_rational_clip = "asymmetric_rational_clip"
    dpskv4_clip = "dpskv4_clip"
    dpskv4_clip_situ = "dpskv4_clip_situ"


class SwiGLUChannelControlScope(StrEnum):
    """SwiGLU channels to which the selected control function is applied."""

    both = "both"
    gate = "gate"
    up = "up"
    none = "none"


class MoSENonlinearity(StrEnum):
    """Function applied to a MoSE nonlinear U projection."""

    silu = "silu"
    rms_norm = "rms_norm"


DPSKV4_CLIP_BOUND = 10.0
DEFAULT_RMS_BETA_SCALE = 4.0


def _validate_rms_beta_scale(value: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise OLMoConfigurationError("rms_beta_scale must be a finite positive number")
    return float(value)


def _rms_situ_up(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Apply SiTU with a per-token beta equal to ``scale * RMS(x)``."""
    rms = x.square().mean(dim=-1, keepdim=True)
    rms = rms.clamp_min(torch.finfo(x.dtype).tiny).sqrt()
    beta = scale * rms
    return beta * torch.tanh(x / beta)


def _apply_mose_nonlinearity(
    x: torch.Tensor,
    nonlinearity: MoSENonlinearity,
    *,
    rms_norm: Optional[RMSNorm] = None,
) -> torch.Tensor:
    if nonlinearity == MoSENonlinearity.silu:
        return F.silu(x)
    if nonlinearity == MoSENonlinearity.rms_norm:
        if rms_norm is None:
            raise RuntimeError("RMSNorm nonlinearity requires an RMSNorm module")
        return rms_norm(x)
    raise NotImplementedError(nonlinearity)


def _rational_clip(x: torch.Tensor, beta: float) -> torch.Tensor:
    """Evaluate ``x * rsqrt(1 + (x / beta)^2)`` without overflow."""
    abs_x = x.abs()
    direct_x = x.clamp(min=-beta, max=beta)
    direct = direct_x * torch.rsqrt(1.0 + (direct_x / beta).square())

    safe_abs_x = abs_x.clamp_min(beta)
    reciprocal = x.sign() * beta * torch.rsqrt(1.0 + (beta / safe_abs_x).square())
    return torch.where(abs_x <= beta, direct, reciprocal)


def _apply_swiglu_channel_control(
    gate: torch.Tensor,
    up: torch.Tensor,
    *,
    control: SwiGLUChannelControl,
    control_scope: SwiGLUChannelControlScope,
    beta_gate: float,
    beta_up: float,
    rms_beta_scale: float = DEFAULT_RMS_BETA_SCALE,
) -> torch.Tensor:
    if (
        control == SwiGLUChannelControl.standard
        or control_scope == SwiGLUChannelControlScope.none
    ):
        return F.silu(gate) * up

    projection_dtype = gate.dtype
    if projection_dtype in (torch.float16, torch.bfloat16):
        gate = gate.float()
        up = up.float()

    control_gate = control_scope in (
        SwiGLUChannelControlScope.both,
        SwiGLUChannelControlScope.gate,
    )
    control_up = control_scope in (
        SwiGLUChannelControlScope.both,
        SwiGLUChannelControlScope.up,
    )

    if control in (
        SwiGLUChannelControl.situ,
        SwiGLUChannelControl.situ_rms,
    ):
        gate_factor = F.silu(gate)
        if control_gate:
            gate_linear = beta_gate * torch.tanh(gate / beta_gate)
            gate_factor = gate_linear * torch.sigmoid(gate)
        if control_up:
            if control == SwiGLUChannelControl.situ:
                up = beta_up * torch.tanh(up / beta_up)
            else:
                up = _rms_situ_up(up, rms_beta_scale)
    elif control == SwiGLUChannelControl.asymmetric_rational_clip:
        gate_factor = F.silu(gate)
        if control_up:
            up = _rational_clip(up, beta_up)
        if control_gate:
            gate_linear = torch.where(gate > 0, _rational_clip(gate, beta_gate), gate)
            # The negative-tail limit of gate * sigmoid(gate) is zero, not inf * 0.
            gate_linear = torch.where(
                torch.isneginf(gate_linear),
                torch.zeros_like(gate_linear),
                gate_linear,
            )
            gate_factor = gate_linear * torch.sigmoid(gate)
    elif control in (
        SwiGLUChannelControl.dpskv4_clip,
        SwiGLUChannelControl.dpskv4_clip_situ,
    ):
        if control_gate:
            gate = gate.clamp_max(DPSKV4_CLIP_BOUND)
        gate_factor = F.silu(gate)
        if control_up:
            if control == SwiGLUChannelControl.dpskv4_clip:
                up = up.clamp(min=-DPSKV4_CLIP_BOUND, max=DPSKV4_CLIP_BOUND)
            else:
                up = beta_up * torch.tanh(up / beta_up)
    else:
        raise NotImplementedError(control)

    return (gate_factor * up).to(projection_dtype)


@dataclass
class ChannelControlledFeedForwardConfig(FeedForwardConfig):
    """OLMo feed-forward config with fixed-function SwiGLU channel control."""

    control: SwiGLUChannelControl = SwiGLUChannelControl.situ
    control_scope: SwiGLUChannelControlScope = SwiGLUChannelControlScope.both
    rms_beta_scale: float = DEFAULT_RMS_BETA_SCALE

    def __post_init__(self) -> None:
        self.name = FeedForwardType(self.name)
        self.activation = ActivationFunction(self.activation)
        self.control = SwiGLUChannelControl(self.control)
        self.control_scope = SwiGLUChannelControlScope(self.control_scope)
        self.rms_beta_scale = _validate_rms_beta_scale(self.rms_beta_scale)

        if self.name != FeedForwardType.default:
            raise OLMoConfigurationError(
                "channel-controlled SwiGLU requires the default feed-forward implementation"
            )
        if self.activation != ActivationFunction.silu:
            raise OLMoConfigurationError("channel control is only defined for SwiGLU")

    def build(
        self,
        d_model: int,
        *,
        dtype: Optional[torch.dtype] = None,
        init_device: str = "cpu",
    ) -> "ChannelControlledFeedForward":
        kwargs = self.as_dict(exclude_none=True)
        kwargs.pop("name")
        kwargs.update(d_model=d_model, init_device=init_device)
        if self.dtype is not None:
            kwargs["dtype"] = self.dtype.as_pt()
        elif dtype is not None:
            kwargs["dtype"] = dtype

        try:
            return ChannelControlledFeedForward(**kwargs)
        except TypeError as e:
            raise OLMoConfigurationError(
                f"invalid options for {self.__class__.__name__}, {e}"
            ) from e


class ChannelControlledFeedForward(FeedForward):
    """SwiGLU with configurable fixed-function gate and up channel control."""

    def __init__(
        self,
        *,
        d_model: int,
        hidden_size: int,
        bias: bool = True,
        dtype: torch.dtype = torch.float32,
        init_device: str = "cpu",
        activation: ActivationFunction = ActivationFunction.silu,
        control: SwiGLUChannelControl = SwiGLUChannelControl.situ,
        control_scope: SwiGLUChannelControlScope = SwiGLUChannelControlScope.both,
        rms_beta_scale: float = DEFAULT_RMS_BETA_SCALE,
    ):
        activation = ActivationFunction(activation)
        if activation != ActivationFunction.silu:
            raise OLMoConfigurationError("channel control is only defined for SwiGLU")

        super().__init__(
            d_model=d_model,
            hidden_size=hidden_size,
            bias=bias,
            dtype=dtype,
            init_device=init_device,
            activation=activation,
        )
        self.control = SwiGLUChannelControl(control)
        self.control_scope = SwiGLUChannelControlScope(control_scope)
        self.rms_beta_scale = _validate_rms_beta_scale(rms_beta_scale)
        self.beta_gate = 4.0
        self.beta_up = 25.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.w1(x)
        up = self.w3(x)
        hidden = _apply_swiglu_channel_control(
            gate,
            up,
            control=self.control,
            control_scope=self.control_scope,
            beta_gate=self.beta_gate,
            beta_up=self.beta_up,
            rms_beta_scale=self.rms_beta_scale,
        )
        return self.w2(hidden)


def _validate_mose_ranks(r1: int, r2: int, down_r1: int, down_r2: int) -> None:
    ranks = {
        "r1": r1,
        "r2": r2,
        "down_r1": down_r1,
        "down_r2": down_r2,
    }
    for name, rank in ranks.items():
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
            raise OLMoConfigurationError(f"{name} must be a non-negative integer")
    if r1 == 0 and r2 == 0:
        raise OLMoConfigurationError("at least one of r1 or r2 must be greater than zero")


@dataclass
class MoSESwiGLUConfig(FeedForwardConfig):
    """Configuration for a Mixture of Subspace Experts SwiGLU."""

    bias: Optional[bool] = False
    r1: int = 880
    r2: int = 880
    down_r1: int = 880
    down_r2: int = 880
    control: SwiGLUChannelControl = SwiGLUChannelControl.situ
    control_scope: SwiGLUChannelControlScope = SwiGLUChannelControlScope.both
    rms_beta_scale: float = DEFAULT_RMS_BETA_SCALE
    gate_nonlinearity: MoSENonlinearity = MoSENonlinearity.silu
    up_nonlinearity: MoSENonlinearity = MoSENonlinearity.silu
    down_nonlinearity: MoSENonlinearity = MoSENonlinearity.silu
    rms_norm_learnable_weight: bool = False
    share_gate_up_subspace: bool = True

    def __post_init__(self) -> None:
        self.name = FeedForwardType(self.name)
        self.activation = ActivationFunction(self.activation)
        self.control = SwiGLUChannelControl(self.control)
        self.control_scope = SwiGLUChannelControlScope(self.control_scope)
        self.rms_beta_scale = _validate_rms_beta_scale(self.rms_beta_scale)
        self.gate_nonlinearity = MoSENonlinearity(self.gate_nonlinearity)
        self.up_nonlinearity = MoSENonlinearity(self.up_nonlinearity)
        self.down_nonlinearity = MoSENonlinearity(self.down_nonlinearity)
        if not isinstance(self.rms_norm_learnable_weight, bool):
            raise OLMoConfigurationError("rms_norm_learnable_weight must be a boolean")
        if not isinstance(self.share_gate_up_subspace, bool):
            raise OLMoConfigurationError("share_gate_up_subspace must be a boolean")

        if self.name != FeedForwardType.default:
            raise OLMoConfigurationError(
                "MoSE-SwiGLU requires the default feed-forward implementation"
            )
        if self.activation != ActivationFunction.silu:
            raise OLMoConfigurationError("MoSE channel control is only defined for SwiGLU")
        if self.bias is None:
            self.bias = False
        _validate_mose_ranks(self.r1, self.r2, self.down_r1, self.down_r2)

    def num_params(self, d_model: int) -> int:
        hidden_size = self.hidden_size
        gate_up_u_count = 1 if self.share_gate_up_subspace else 2
        params = (self.r1 + self.r2) * (
            gate_up_u_count * d_model + 2 * hidden_size
        )
        if self.bias:
            params += 2 * hidden_size

        if self.down_r1 == 0 and self.down_r2 == 0:
            params += hidden_size * d_model
        else:
            params += (self.down_r1 + self.down_r2) * (hidden_size + d_model)
        if self.bias:
            params += d_model
        if self.rms_norm_learnable_weight:
            if self.r2 > 0:
                if self.share_gate_up_subspace:
                    if MoSENonlinearity.rms_norm in (
                        self.gate_nonlinearity,
                        self.up_nonlinearity,
                    ):
                        params += self.r2
                else:
                    if self.gate_nonlinearity == MoSENonlinearity.rms_norm:
                        params += self.r2
                    if self.up_nonlinearity == MoSENonlinearity.rms_norm:
                        params += self.r2
            if self.down_r2 > 0 and self.down_nonlinearity == MoSENonlinearity.rms_norm:
                params += self.down_r2
        return params

    def build(
        self,
        d_model: int,
        *,
        dtype: Optional[torch.dtype] = None,
        init_device: str = "cpu",
    ) -> "MoSESwiGLU":
        kwargs = self.as_dict(exclude_none=True)
        kwargs.pop("name")
        kwargs.update(d_model=d_model, init_device=init_device)
        if self.dtype is not None:
            kwargs["dtype"] = self.dtype.as_pt()
        elif dtype is not None:
            kwargs["dtype"] = dtype

        try:
            return MoSESwiGLU(**kwargs)
        except TypeError as e:
            raise OLMoConfigurationError(
                f"invalid options for {self.__class__.__name__}, {e}"
            ) from e


class MoSESwiGLU(nn.Module):
    """Mixture of linear and nonlinear low-rank subspace experts for SwiGLU."""

    def __init__(
        self,
        *,
        d_model: int,
        hidden_size: int,
        r1: int = 880,
        r2: int = 880,
        down_r1: int = 880,
        down_r2: int = 880,
        bias: bool = False,
        dtype: torch.dtype = torch.float32,
        init_device: str = "cpu",
        activation: ActivationFunction = ActivationFunction.silu,
        control: SwiGLUChannelControl = SwiGLUChannelControl.situ,
        control_scope: SwiGLUChannelControlScope = SwiGLUChannelControlScope.both,
        rms_beta_scale: float = DEFAULT_RMS_BETA_SCALE,
        gate_nonlinearity: MoSENonlinearity = MoSENonlinearity.silu,
        up_nonlinearity: MoSENonlinearity = MoSENonlinearity.silu,
        down_nonlinearity: MoSENonlinearity = MoSENonlinearity.silu,
        rms_norm_learnable_weight: bool = False,
        share_gate_up_subspace: bool = True,
    ):
        super().__init__()
        activation = ActivationFunction(activation)
        if activation != ActivationFunction.silu:
            raise OLMoConfigurationError("MoSE channel control is only defined for SwiGLU")
        _validate_mose_ranks(r1, r2, down_r1, down_r2)

        self.d_model = d_model
        self.hidden_size = hidden_size
        self.r1 = r1
        self.r2 = r2
        self.down_r1 = down_r1
        self.down_r2 = down_r2
        self.control = SwiGLUChannelControl(control)
        self.control_scope = SwiGLUChannelControlScope(control_scope)
        self.rms_beta_scale = _validate_rms_beta_scale(rms_beta_scale)
        self.gate_nonlinearity = MoSENonlinearity(gate_nonlinearity)
        self.up_nonlinearity = MoSENonlinearity(up_nonlinearity)
        self.down_nonlinearity = MoSENonlinearity(down_nonlinearity)
        if not isinstance(rms_norm_learnable_weight, bool):
            raise OLMoConfigurationError("rms_norm_learnable_weight must be a boolean")
        if not isinstance(share_gate_up_subspace, bool):
            raise OLMoConfigurationError("share_gate_up_subspace must be a boolean")
        self.rms_norm_learnable_weight = rms_norm_learnable_weight
        self.share_gate_up_subspace = share_gate_up_subspace
        self.beta_gate = 4.0
        self.beta_up = 25.0

        linear_kwargs = {"bias": False, "dtype": dtype, "device": init_device}

        self.linear_u = None
        self.gate_linear_u = None
        self.up_linear_u = None
        if r1 > 0:
            if share_gate_up_subspace:
                self.linear_u = nn.Linear(d_model, r1, **linear_kwargs)
            else:
                self.gate_linear_u = nn.Linear(d_model, r1, **linear_kwargs)
                self.up_linear_u = nn.Linear(d_model, r1, **linear_kwargs)
            self.gate_linear_v = nn.Linear(r1, hidden_size, **linear_kwargs)
            self.up_linear_v = nn.Linear(r1, hidden_size, **linear_kwargs)
        else:
            self.gate_linear_v = None
            self.up_linear_v = None

        self.nonlinear_u = None
        self.gate_nonlinear_u = None
        self.up_nonlinear_u = None
        self.gate_up_nonlinear_norm = None
        self.gate_nonlinear_norm = None
        self.up_nonlinear_norm = None
        if r2 > 0:
            if share_gate_up_subspace:
                self.nonlinear_u = nn.Linear(d_model, r2, **linear_kwargs)
            else:
                self.gate_nonlinear_u = nn.Linear(d_model, r2, **linear_kwargs)
                self.up_nonlinear_u = nn.Linear(d_model, r2, **linear_kwargs)
            self.gate_nonlinear_v = nn.Linear(r2, hidden_size, **linear_kwargs)
            self.up_nonlinear_v = nn.Linear(r2, hidden_size, **linear_kwargs)
            if share_gate_up_subspace:
                if MoSENonlinearity.rms_norm in (
                    self.gate_nonlinearity,
                    self.up_nonlinearity,
                ):
                    self.gate_up_nonlinear_norm = RMSNorm(
                        size=r2,
                        eps=1e-5,
                        elementwise_affine=rms_norm_learnable_weight,
                        bias=False,
                        dtype=dtype,
                        init_device=init_device,
                    )
            else:
                if self.gate_nonlinearity == MoSENonlinearity.rms_norm:
                    self.gate_nonlinear_norm = RMSNorm(
                        size=r2,
                        eps=1e-5,
                        elementwise_affine=rms_norm_learnable_weight,
                        bias=False,
                        dtype=dtype,
                        init_device=init_device,
                    )
                if self.up_nonlinearity == MoSENonlinearity.rms_norm:
                    self.up_nonlinear_norm = RMSNorm(
                        size=r2,
                        eps=1e-5,
                        elementwise_affine=rms_norm_learnable_weight,
                        bias=False,
                        dtype=dtype,
                        init_device=init_device,
                    )
        else:
            self.gate_nonlinear_v = None
            self.up_nonlinear_v = None

        if bias:
            self.gate_bias = nn.Parameter(
                torch.zeros(hidden_size, dtype=dtype, device=init_device)
            )
            self.up_bias = nn.Parameter(
                torch.zeros(hidden_size, dtype=dtype, device=init_device)
            )
        else:
            self.register_parameter("gate_bias", None)
            self.register_parameter("up_bias", None)

        self.down_is_mose = down_r1 > 0 or down_r2 > 0
        if not self.down_is_mose:
            self.w_down = nn.Linear(
                hidden_size,
                d_model,
                bias=bias,
                dtype=dtype,
                device=init_device,
            )
            self.down_linear_u = None
            self.down_linear_v = None
            self.down_nonlinear_u = None
            self.down_nonlinear_v = None
            self.down_nonlinear_norm = None
            self.register_parameter("down_bias", None)
        else:
            self.w_down = None
            if down_r1 > 0:
                self.down_linear_u = nn.Linear(hidden_size, down_r1, **linear_kwargs)
                self.down_linear_v = nn.Linear(down_r1, d_model, **linear_kwargs)
            else:
                self.down_linear_u = None
                self.down_linear_v = None

            if down_r2 > 0:
                self.down_nonlinear_u = nn.Linear(hidden_size, down_r2, **linear_kwargs)
                self.down_nonlinear_v = nn.Linear(down_r2, d_model, **linear_kwargs)
                if self.down_nonlinearity == MoSENonlinearity.rms_norm:
                    self.down_nonlinear_norm = RMSNorm(
                        size=down_r2,
                        eps=1e-5,
                        elementwise_affine=rms_norm_learnable_weight,
                        bias=False,
                        dtype=dtype,
                        init_device=init_device,
                    )
                else:
                    self.down_nonlinear_norm = None
            else:
                self.down_nonlinear_u = None
                self.down_nonlinear_v = None
                self.down_nonlinear_norm = None

            if bias:
                self.down_bias = nn.Parameter(
                    torch.zeros(d_model, dtype=dtype, device=init_device)
                )
            else:
                self.register_parameter("down_bias", None)

    def projection_modules(self) -> tuple[nn.Module, ...]:
        """Return projections in their deterministic OLMo initialization order."""
        modules = [
            self.linear_u,
            self.gate_linear_u,
            self.up_linear_u,
            self.gate_linear_v,
            self.up_linear_v,
            self.nonlinear_u,
            self.gate_nonlinear_u,
            self.up_nonlinear_u,
            self.gate_nonlinear_v,
            self.up_nonlinear_v,
            self.w_down,
            self.down_linear_u,
            self.down_linear_v,
            self.down_nonlinear_u,
            self.down_nonlinear_v,
        ]
        return tuple(module for module in modules if module is not None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = None
        up = None

        if self.linear_u is not None:
            linear_gate_latent = self.linear_u(x)
            linear_up_latent = linear_gate_latent
        elif self.gate_linear_u is not None:
            assert self.up_linear_u is not None
            linear_gate_latent = self.gate_linear_u(x)
            linear_up_latent = self.up_linear_u(x)
        else:
            linear_gate_latent = None
            linear_up_latent = None

        if linear_gate_latent is not None:
            assert self.gate_linear_v is not None and self.up_linear_v is not None
            assert linear_up_latent is not None
            gate = self.gate_linear_v(linear_gate_latent)
            up = self.up_linear_v(linear_up_latent)

        if self.nonlinear_u is not None:
            gate_nonlinear_input = self.nonlinear_u(x)
            up_nonlinear_input = gate_nonlinear_input
            gate_nonlinear_norm = self.gate_up_nonlinear_norm
            up_nonlinear_norm = self.gate_up_nonlinear_norm
        elif self.gate_nonlinear_u is not None:
            assert self.up_nonlinear_u is not None
            gate_nonlinear_input = self.gate_nonlinear_u(x)
            up_nonlinear_input = self.up_nonlinear_u(x)
            gate_nonlinear_norm = self.gate_nonlinear_norm
            up_nonlinear_norm = self.up_nonlinear_norm
        else:
            gate_nonlinear_input = None
            up_nonlinear_input = None

        if gate_nonlinear_input is not None:
            assert up_nonlinear_input is not None
            gate_nonlinear_latent = _apply_mose_nonlinearity(
                gate_nonlinear_input,
                self.gate_nonlinearity,
                rms_norm=gate_nonlinear_norm,
            )
            if (
                self.share_gate_up_subspace
                and self.up_nonlinearity == self.gate_nonlinearity
            ):
                up_nonlinear_latent = gate_nonlinear_latent
            else:
                up_nonlinear_latent = _apply_mose_nonlinearity(
                    up_nonlinear_input,
                    self.up_nonlinearity,
                    rms_norm=up_nonlinear_norm,
                )
            assert self.gate_nonlinear_v is not None and self.up_nonlinear_v is not None
            gate_nonlinear = self.gate_nonlinear_v(gate_nonlinear_latent)
            up_nonlinear = self.up_nonlinear_v(up_nonlinear_latent)
            gate = gate_nonlinear if gate is None else gate + gate_nonlinear
            up = up_nonlinear if up is None else up + up_nonlinear

        assert gate is not None and up is not None
        if self.gate_bias is not None:
            gate = gate + self.gate_bias
            up = up + self.up_bias

        hidden = _apply_swiglu_channel_control(
            gate,
            up,
            control=self.control,
            control_scope=self.control_scope,
            beta_gate=self.beta_gate,
            beta_up=self.beta_up,
            rms_beta_scale=self.rms_beta_scale,
        )

        if not self.down_is_mose:
            assert self.w_down is not None
            return self.w_down(hidden)

        out = None
        if self.down_linear_u is not None:
            assert self.down_linear_v is not None
            out = self.down_linear_v(self.down_linear_u(hidden))
        if self.down_nonlinear_u is not None:
            assert self.down_nonlinear_v is not None
            down_nonlinear_latent = _apply_mose_nonlinearity(
                self.down_nonlinear_u(hidden),
                self.down_nonlinearity,
                rms_norm=self.down_nonlinear_norm,
            )
            nonlinear_out = self.down_nonlinear_v(down_nonlinear_latent)
            out = nonlinear_out if out is None else out + nonlinear_out

        assert out is not None
        if self.down_bias is not None:
            out = out + self.down_bias
        return out

    def apply_tp(
        self,
        tp_mesh: DeviceMesh,
        input_layout: Optional[Placement] = None,
        output_layout: Optional[Placement] = None,
        use_local_output: bool = True,
        float8_enabled: bool = False,
    ) -> None:
        del tp_mesh, input_layout, output_layout, use_local_output, float8_enabled
        raise NotImplementedError("tensor parallelism is not implemented for MoSE-SwiGLU")

    def num_flops_per_token(self, seq_len: int) -> int:
        del seq_len
        return 6 * sum(parameter.numel() for parameter in self.parameters())
