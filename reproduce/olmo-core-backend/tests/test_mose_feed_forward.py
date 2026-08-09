import pytest
import torch
import torch.nn.functional as F

from olmo_core.exceptions import OLMoConfigurationError
from olmo_mose import MoSESwiGLU, MoSESwiGLUConfig, SwiGLUChannelControl


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
        latent = F.silu(module.nonlinear_u(x))
        nonlinear_gate = module.gate_nonlinear_v(latent)
        nonlinear_up = module.up_nonlinear_v(latent)
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
        nonlinear_out = module.down_nonlinear_v(F.silu(module.down_nonlinear_u(hidden)))
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
