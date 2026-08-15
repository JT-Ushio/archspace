import math
from typing import Optional

import pytest
import torch
import torch.nn.functional as F

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.transformer.init import InitMethod
from olmo_core.nn.layer_norm import RMSNorm
from olmo_mose import (
    MoSENonlinearity,
    MoSESwiGLU,
    MoSESwiGLUConfig,
    SwiGLUChannelControl,
)
from olmo_mose import hooks


def _apply_nonlinearity(
    x: torch.Tensor,
    nonlinearity: MoSENonlinearity,
    *,
    rms_norm: Optional[RMSNorm] = None,
) -> torch.Tensor:
    if nonlinearity == MoSENonlinearity.silu:
        return F.silu(x)
    if nonlinearity == MoSENonlinearity.rms_norm:
        assert rms_norm is not None
        input_dtype = x.dtype
        x = x.float()
        variance = x.square().mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + rms_norm.eps)
        if rms_norm.weight is not None:
            x = rms_norm.weight.type_as(x) * x
        return x.to(input_dtype)
    raise NotImplementedError(nonlinearity)


def _controlled_hidden(
    gate: torch.Tensor,
    up: torch.Tensor,
    control: SwiGLUChannelControl,
) -> torch.Tensor:
    if control == SwiGLUChannelControl.standard:
        return F.silu(gate) * up
    if control == SwiGLUChannelControl.situ:
        gate_linear = 4.0 * torch.tanh(gate / 4.0)
        up = 25.0 * torch.tanh(up / 25.0)
        return gate_linear * torch.sigmoid(gate) * up
    if control == SwiGLUChannelControl.asymmetric_rational_clip:
        up = up * torch.rsqrt(1.0 + (up / 25.0).square())
        gate_linear = gate * torch.rsqrt(1.0 + (F.relu(gate) / 4.0).square())
        return gate_linear * torch.sigmoid(gate) * up
    raise NotImplementedError(control)


def _reference_forward(module: MoSESwiGLU, x: torch.Tensor) -> torch.Tensor:
    gate = None
    up = None
    if module.linear_u is not None:
        latent = module.linear_u(x)
        gate = module.gate_linear_v(latent)
        up = module.up_linear_v(latent)
    if module.nonlinear_u is not None:
        latent = module.nonlinear_u(x)
        nonlinear_gate = module.gate_nonlinear_v(
            _apply_nonlinearity(
                latent,
                module.gate_nonlinearity,
                rms_norm=module.gate_up_nonlinear_norm,
            )
        )
        nonlinear_up = module.up_nonlinear_v(
            _apply_nonlinearity(
                latent,
                module.up_nonlinearity,
                rms_norm=module.gate_up_nonlinear_norm,
            )
        )
        gate = nonlinear_gate if gate is None else gate + nonlinear_gate
        up = nonlinear_up if up is None else up + nonlinear_up

    assert gate is not None and up is not None
    if module.gate_bias is not None:
        gate = gate + module.gate_bias
        up = up + module.up_bias
    hidden = _controlled_hidden(gate, up, module.control)

    if not module.down_is_mose:
        return module.w_down(hidden)

    out = None
    if module.down_linear_u is not None:
        out = module.down_linear_v(module.down_linear_u(hidden))
    if module.down_nonlinear_u is not None:
        nonlinear_out = module.down_nonlinear_v(
            _apply_nonlinearity(
                module.down_nonlinear_u(hidden),
                module.down_nonlinearity,
                rms_norm=module.down_nonlinear_norm,
            )
        )
        out = nonlinear_out if out is None else out + nonlinear_out
    assert out is not None
    if module.down_bias is not None:
        out = out + module.down_bias
    return out


@pytest.mark.parametrize("control", list(SwiGLUChannelControl))
@pytest.mark.parametrize("ranks", [(3, 2), (3, 0), (0, 2)])
@pytest.mark.parametrize("down_ranks", [(0, 0), (3, 2), (3, 0), (0, 2)])
def test_mose_forward_matches_reference(
    control: SwiGLUChannelControl,
    ranks: tuple[int, int],
    down_ranks: tuple[int, int],
) -> None:
    torch.manual_seed(10)
    module = MoSESwiGLU(
        d_model=4,
        hidden_size=7,
        r1=ranks[0],
        r2=ranks[1],
        down_r1=down_ranks[0],
        down_r2=down_ranks[1],
        bias=True,
        dtype=torch.float64,
        control=control,
    )
    with torch.no_grad():
        module.gate_bias.fill_(0.25)
        module.up_bias.fill_(-0.5)
        if module.down_bias is not None:
            module.down_bias.fill_(0.75)
    x = torch.randn(2, 3, 4, dtype=torch.float64)

    torch.testing.assert_close(module(x), _reference_forward(module, x))


@pytest.mark.parametrize(
    ("gate_nonlinearity", "up_nonlinearity", "down_nonlinearity"),
    [
        (MoSENonlinearity.silu, MoSENonlinearity.rms_norm, MoSENonlinearity.rms_norm),
        (MoSENonlinearity.rms_norm, MoSENonlinearity.rms_norm, MoSENonlinearity.silu),
    ],
)
def test_mose_configurable_nonlinearities_match_reference(
    gate_nonlinearity: MoSENonlinearity,
    up_nonlinearity: MoSENonlinearity,
    down_nonlinearity: MoSENonlinearity,
) -> None:
    torch.manual_seed(11)
    module = MoSESwiGLU(
        d_model=4,
        hidden_size=7,
        r1=3,
        r2=2,
        down_r1=3,
        down_r2=2,
        dtype=torch.float64,
        control=SwiGLUChannelControl.standard,
        gate_nonlinearity=gate_nonlinearity,
        up_nonlinearity=up_nonlinearity,
        down_nonlinearity=down_nonlinearity,
    )
    x = torch.randn(2, 3, 4, dtype=torch.float64)

    torch.testing.assert_close(module(x), _reference_forward(module, x))


@pytest.mark.parametrize("learnable_weight", [False, True])
def test_mose_rms_norm_weight_is_optional_and_shared_for_gate_up(
    learnable_weight: bool,
) -> None:
    torch.manual_seed(12)
    config = MoSESwiGLUConfig(
        hidden_size=7,
        r1=3,
        r2=2,
        down_r1=4,
        down_r2=3,
        gate_nonlinearity=MoSENonlinearity.rms_norm,
        up_nonlinearity=MoSENonlinearity.rms_norm,
        down_nonlinearity=MoSENonlinearity.rms_norm,
        rms_norm_learnable_weight=learnable_weight,
    )
    module = config.build(d_model=4, dtype=torch.float64)

    assert module.gate_up_nonlinear_norm is not None
    assert module.down_nonlinear_norm is not None
    if learnable_weight:
        assert module.gate_up_nonlinear_norm.weight is not None
        assert module.gate_up_nonlinear_norm.weight.shape == (2,)
        assert module.down_nonlinear_norm.weight is not None
        assert module.down_nonlinear_norm.weight.shape == (3,)
        with torch.no_grad():
            module.gate_up_nonlinear_norm.weight.copy_(torch.tensor([0.5, 1.5]))
            module.down_nonlinear_norm.weight.copy_(torch.tensor([0.5, 1.0, 1.5]))
        assert "gate_up_nonlinear_norm.weight" in module.state_dict()
        assert "down_nonlinear_norm.weight" in module.state_dict()
    else:
        assert module.gate_up_nonlinear_norm.weight is None
        assert module.down_nonlinear_norm.weight is None
        assert "gate_up_nonlinear_norm.weight" not in module.state_dict()
        assert "down_nonlinear_norm.weight" not in module.state_dict()

    x = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)
    torch.testing.assert_close(module(x), _reference_forward(module, x))
    module(x).sum().backward()
    if learnable_weight:
        assert module.gate_up_nonlinear_norm.weight.grad is not None
        assert module.down_nonlinear_norm.weight.grad is not None

    expected_extra_params = 5 if learnable_weight else 0
    base_config = MoSESwiGLUConfig(
        hidden_size=7,
        r1=3,
        r2=2,
        down_r1=4,
        down_r2=3,
        gate_nonlinearity=MoSENonlinearity.rms_norm,
        up_nonlinearity=MoSENonlinearity.rms_norm,
        down_nonlinearity=MoSENonlinearity.rms_norm,
    )
    assert config.num_params(4) == base_config.num_params(4) + expected_extra_params
    assert config.num_params(4) == sum(parameter.numel() for parameter in module.parameters())


@pytest.mark.parametrize(
    ("r1", "r2", "down_r1", "down_r2"),
    [(3, 0, 2, 0), (0, 3, 0, 2), (3, 0, 0, 2), (0, 3, 2, 0)],
)
def test_mose_supports_individually_disabled_experts(
    r1: int,
    r2: int,
    down_r1: int,
    down_r2: int,
) -> None:
    module = MoSESwiGLU(
        d_model=4,
        hidden_size=6,
        r1=r1,
        r2=r2,
        down_r1=down_r1,
        down_r2=down_r2,
        control=SwiGLUChannelControl.situ,
    )
    x = torch.randn(2, 4, requires_grad=True)

    module(x).sum().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert all(parameter.grad is not None for parameter in module.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in module.parameters())


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("ranks", [(3, 2), (3, 0), (0, 2)])
@pytest.mark.parametrize("down_ranks", [(0, 0), (3, 2), (3, 0), (0, 2)])
def test_mose_config_parameter_count_matches_module(
    bias: bool,
    ranks: tuple[int, int],
    down_ranks: tuple[int, int],
) -> None:
    config = MoSESwiGLUConfig(
        hidden_size=7,
        r1=ranks[0],
        r2=ranks[1],
        down_r1=down_ranks[0],
        down_r2=down_ranks[1],
        bias=bias,
    )
    module = config.build(d_model=4)

    assert config.num_params(4) == sum(parameter.numel() for parameter in module.parameters())


def test_mose_default_rank_and_checkpoint_keys() -> None:
    config = MoSESwiGLUConfig(hidden_size=16, bias=False)
    module = config.build(d_model=8)

    assert (config.r1, config.r2, config.down_r1, config.down_r2) == (880, 880, 880, 880)
    assert set(module.state_dict()) == {
        "linear_u.weight",
        "gate_linear_v.weight",
        "up_linear_v.weight",
        "nonlinear_u.weight",
        "gate_nonlinear_v.weight",
        "up_nonlinear_v.weight",
        "down_linear_u.weight",
        "down_linear_v.weight",
        "down_nonlinear_u.weight",
        "down_nonlinear_v.weight",
    }


def test_mose_defaults_preserve_silu_nonlinearities() -> None:
    config = MoSESwiGLUConfig(hidden_size=16)

    assert config.gate_nonlinearity == MoSENonlinearity.silu
    assert config.up_nonlinearity == MoSENonlinearity.silu
    assert config.down_nonlinearity == MoSENonlinearity.silu
    assert config.rms_norm_learnable_weight is False


def test_mose_initializes_uv_factors_for_target_matrix_std(monkeypatch) -> None:
    module = MoSESwiGLU(
        d_model=4,
        hidden_size=7,
        r1=3,
        r2=2,
        down_r1=4,
        down_r2=3,
    )
    calls = {}

    def record_init(projection, *, std, generator):
        del generator
        calls[id(projection)] = std

    monkeypatch.setattr(hooks, "init_linear", record_init)
    InitMethod.normal.init_feed_forward(
        module,
        d_model=4,
        block_idx=0,
        num_blocks=1,
        std=0.02,
    )

    for projection in (module.linear_u, module.nonlinear_u):
        assert calls[id(projection)] == 0.02
    for projection in (
        module.gate_linear_v,
        module.up_linear_v,
        module.gate_nonlinear_v,
        module.up_nonlinear_v,
    ):
        assert calls[id(projection)] == pytest.approx(1.0 / math.sqrt(5))
    for projection in (module.down_linear_u, module.down_nonlinear_u):
        assert calls[id(projection)] == 0.02
    for projection in (module.down_linear_v, module.down_nonlinear_v):
        assert calls[id(projection)] == pytest.approx(1.0 / math.sqrt(7))


def test_mose_controls_share_checkpoint_topology() -> None:
    situ = MoSESwiGLU(
        d_model=4,
        hidden_size=6,
        r1=3,
        r2=2,
        down_r1=3,
        down_r2=2,
        control=SwiGLUChannelControl.situ,
    )
    rational_clip = MoSESwiGLU(
        d_model=4,
        hidden_size=6,
        r1=3,
        r2=2,
        down_r1=3,
        down_r2=2,
        control=SwiGLUChannelControl.asymmetric_rational_clip,
    )

    rational_clip.load_state_dict(situ.state_dict(), strict=True)


@pytest.mark.parametrize(
    ("control", "gate_value", "up_value"),
    [
        (SwiGLUChannelControl.situ, 14.0, 87.5),
        (SwiGLUChannelControl.asymmetric_rational_clip, 100.0, 338.0),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_mose_reuses_fp32_channel_control(
    control: SwiGLUChannelControl,
    gate_value: float,
    up_value: float,
    dtype: torch.dtype,
) -> None:
    module = MoSESwiGLU(
        d_model=1,
        hidden_size=1,
        r1=1,
        r2=0,
        down_r1=0,
        down_r2=0,
        bias=True,
        dtype=dtype,
        control=control,
    )
    with torch.no_grad():
        module.linear_u.weight.zero_()
        module.gate_linear_v.weight.zero_()
        module.up_linear_v.weight.zero_()
        module.gate_bias.fill_(gate_value)
        module.up_bias.fill_(up_value)
        module.w_down.weight.fill_(1.0)
        module.w_down.bias.zero_()
    x = torch.zeros(1, 1, dtype=dtype, requires_grad=True)

    output = module(x)
    output.sum().backward()
    gate = torch.tensor(gate_value, dtype=dtype).float()
    up = torch.tensor(up_value, dtype=dtype).float()
    expected = _controlled_hidden(gate, up, control).to(dtype)

    assert output.dtype == dtype
    torch.testing.assert_close(output.squeeze(), expected)
    assert output.item() <= 100.0
    assert module.gate_bias.grad is not None and module.gate_bias.grad.item() > 0
    assert module.up_bias.grad is not None and module.up_bias.grad.item() > 0


def test_mose_compiles_as_a_full_graph() -> None:
    module = MoSESwiGLU(
        d_model=4,
        hidden_size=6,
        r1=3,
        r2=2,
        down_r1=3,
        down_r2=2,
        control=SwiGLUChannelControl.asymmetric_rational_clip,
        gate_nonlinearity=MoSENonlinearity.rms_norm,
        up_nonlinearity=MoSENonlinearity.rms_norm,
        down_nonlinearity=MoSENonlinearity.rms_norm,
        rms_norm_learnable_weight=True,
    )
    compiled = torch.compile(module, backend="eager", fullgraph=True)
    x = torch.randn(2, 4, requires_grad=True)

    compiled(x).sum().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_mose_rejects_tensor_parallelism() -> None:
    module = MoSESwiGLU(d_model=4, hidden_size=6, r1=3, r2=2)

    with pytest.raises(NotImplementedError, match="tensor parallelism"):
        module.apply_tp(None)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("r1", "r2", "down_r1", "down_r2"),
    [(0, 0, 1, 1), (-1, 1, 1, 1), (1, 1, -1, 1), (True, 1, 1, 1)],
)
def test_mose_rejects_invalid_ranks(
    r1: int,
    r2: int,
    down_r1: int,
    down_r2: int,
) -> None:
    with pytest.raises(OLMoConfigurationError):
        MoSESwiGLUConfig(
            hidden_size=8,
            r1=r1,
            r2=r2,
            down_r1=down_r1,
            down_r2=down_r2,
        )


def test_mose_rejects_non_boolean_rms_norm_weight_option() -> None:
    with pytest.raises(OLMoConfigurationError, match="rms_norm_learnable_weight"):
        MoSESwiGLUConfig(
            hidden_size=8,
            rms_norm_learnable_weight=1,  # type: ignore[arg-type]
        )
