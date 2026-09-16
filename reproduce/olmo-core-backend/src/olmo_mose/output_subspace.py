"""Content-conditioned orthogonal Down readout; no dense d_model-square matrix."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from olmo_core.config import StrEnum
from olmo_core.exceptions import OLMoConfigurationError


class DownOutputRotation(StrEnum):
    none = "none"
    static = "static"
    dynamic = "dynamic"
    dynamic_residual = "dynamic_residual"


def validate_output_rotation(mode, groups, a_max, *, down_r1, down_r2,
                             down_nonlinearity, bias, d_model=None):
    mode = DownOutputRotation(mode)
    if isinstance(groups, bool) or not isinstance(groups, int) or groups <= 0:
        raise OLMoConfigurationError("down_rotation_groups must be a positive integer")
    if (isinstance(a_max, bool) or not isinstance(a_max, (int, float))
            or not math.isfinite(a_max) or not 0 < a_max <= 1):
        raise OLMoConfigurationError("down_rotation_a_max must be finite in (0, 1]")
    if mode != DownOutputRotation.none:
        if down_r1 != 0 or down_r2 <= 0 or str(down_nonlinearity) != "identity" or bias:
            raise OLMoConfigurationError(
                "Down rotation requires down_r1=0, down_r2>0, identity Down and no bias")
        if d_model is not None and d_model % (2 * groups):
            raise OLMoConfigurationError("d_model must be divisible by 2 * down_rotation_groups")
    return mode


class GroupedOutputRotation(nn.Module):
    """One Cayley angle per contiguous group, shared by its adjacent channel pairs.

    All controls have G*r parameters. Static uses an all-ones reference latent
    (only G effective degrees of freedom); dynamic modes use token RMS-normalized
    z. The residual mode additionally learns G residual gates and computes
    ``y + tanh(g) * (R(z)y - y)``. Its controller is initialized nonzero while
    ``g=0``, so it starts exactly at identity without creating a dead branch.
    Raw Parameters rather than nn.Linear deliberately avoid Float8 conversion.
    """

    def __init__(self, rank, d_model, groups=32, a_max=0.5, mode="dynamic",
                 *, dtype=torch.float32, device="cpu"):
        super().__init__()
        self.mode = validate_output_rotation(
            mode, groups, a_max, down_r1=0, down_r2=rank,
            down_nonlinearity="identity", bias=False, d_model=d_model)
        if self.mode == DownOutputRotation.none:
            raise OLMoConfigurationError("Disabled rotation must not allocate a controller")
        self.rank, self.d_model, self.groups = rank, d_model, groups
        self.a_max = float(a_max)
        self.weight = nn.Parameter(torch.zeros(groups, rank, dtype=dtype, device=device))
        if self.mode == DownOutputRotation.dynamic_residual:
            self.residual_gate = nn.Parameter(torch.zeros(groups, dtype=dtype, device=device))
        else:
            self.register_parameter("residual_gate", None)
        self.reset_parameters()

    def reset_parameters(self, generator=None):
        """Initialize controls without advancing the model's base RNG stream."""
        # Required after OLMo's meta -> to_empty -> FSDP initialization path.
        if self.weight.is_meta:
            return
        if self.mode == DownOutputRotation.dynamic_residual:
            init_generator = None
            if generator is not None and generator.device.type == self.weight.device.type:
                init_generator = torch.Generator(device=generator.device)
                init_generator.set_state(generator.get_state())
                nn.init.normal_(
                    self.weight, std=1.0 / math.sqrt(self.rank), generator=init_generator
                )
            else:
                devices = [self.weight.device] if self.weight.device.type == "cuda" else []
                with torch.random.fork_rng(devices=devices):
                    nn.init.normal_(self.weight, std=1.0 / math.sqrt(self.rank))
            assert self.residual_gate is not None
            nn.init.zeros_(self.residual_gate)
        else:
            nn.init.zeros_(self.weight)

    def extra_repr(self):
        return (f"mode={self.mode}, rank={self.rank}, d_model={self.d_model}, "
                f"groups={self.groups}, a_max={self.a_max}, "
                f"residual_gate={self.residual_gate is not None}, compute_dtype=float32")

    def coefficients(self, z):
        with torch.autocast(device_type=z.device.type, enabled=False):
            if self.mode == DownOutputRotation.static:
                controller_input = torch.ones(self.rank, dtype=torch.float32, device=z.device)
            else:
                z32 = z.float()
                controller_input = z32 * torch.rsqrt(z32.square().mean(-1, keepdim=True) + 1e-6)
            a = self.a_max * torch.tanh(F.linear(controller_input, self.weight.float()))
            denominator = 1 + a.square()
            return (1 - a.square()) / denominator, (2 * a) / denominator

    def forward(self, y, z):
        with torch.autocast(device_type=y.device.type, enabled=False):
            c, s = self.coefficients(z)
            pairs = y.float().reshape(*y.shape[:-1], self.groups, -1, 2)
            even, odd = pairs[..., 0], pairs[..., 1]
            c, s = c.unsqueeze(-1), s.unsqueeze(-1)
            rotated = torch.stack((c * even - s * odd, s * even + c * odd), dim=-1)
            if self.residual_gate is not None:
                gate = torch.tanh(self.residual_gate.float()).reshape(
                    *((1,) * (pairs.ndim - 3)), self.groups, 1, 1
                )
                rotated = pairs + gate * (rotated - pairs)
            return rotated.reshape_as(y).to(y.dtype)
