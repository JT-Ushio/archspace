import pytest
import torch
import torch.nn.functional as F

from olmo_core.nn.feed_forward import FeedForward
from olmo_mose import (
    ChannelControlledFeedForward,
    ChannelControlledFeedForwardConfig,
    SwiGLUChannelControl,
)
from olmo_mose.feed_forward import _rational_clip


def _reference_hidden(
    gate: torch.Tensor,
    up: torch.Tensor,
    control: SwiGLUChannelControl,
    beta_gate: float,
    beta_up: float,
) -> torch.Tensor:
    if control == SwiGLUChannelControl.standard:
        return F.silu(gate) * up
    if control == SwiGLUChannelControl.situ:
        gate_linear = beta_gate * torch.tanh(gate / beta_gate)
        up = beta_up * torch.tanh(up / beta_up)
        return gate_linear * torch.sigmoid(gate) * up
    if control == SwiGLUChannelControl.asymmetric_rational_clip:
        up = up * torch.rsqrt(1.0 + (up / beta_up).square())
        gate_linear = gate * torch.rsqrt(1.0 + (F.relu(gate) / beta_gate).square())
        return gate_linear * torch.sigmoid(gate) * up
    raise NotImplementedError(control)


@pytest.mark.parametrize("control", list(SwiGLUChannelControl))
def test_forward_matches_reference(control: SwiGLUChannelControl) -> None:
    torch.manual_seed(0)
    module = ChannelControlledFeedForwardConfig(
        hidden_size=7,
        bias=True,
        control=control,
    ).build(d_model=5, dtype=torch.float64)
    x = torch.randn(2, 3, 5, dtype=torch.float64)

    gate = module.w1(x)
    up = module.w3(x)
    expected = module.w2(
        _reference_hidden(gate, up, control, module.beta_gate, module.beta_up)
    )

    torch.testing.assert_close(module(x), expected)


def test_standard_mode_matches_native_swiglu() -> None:
    torch.manual_seed(1)
    native = FeedForward(d_model=4, hidden_size=6, bias=False, dtype=torch.float64)
    controlled = ChannelControlledFeedForward(
        d_model=4,
        hidden_size=6,
        bias=False,
        dtype=torch.float64,
        control=SwiGLUChannelControl.standard,
    )
    controlled.load_state_dict(native.state_dict(), strict=True)
    x = torch.randn(2, 4, dtype=torch.float64)

    torch.testing.assert_close(controlled(x), native(x))


@pytest.mark.parametrize(
    "control",
    [SwiGLUChannelControl.situ, SwiGLUChannelControl.asymmetric_rational_clip],
)
def test_controlled_swiglu_has_finite_gradients(control: SwiGLUChannelControl) -> None:
    torch.manual_seed(2)
    module = ChannelControlledFeedForward(
        d_model=3,
        hidden_size=5,
        bias=False,
        dtype=torch.float64,
        control=control,
    )
    x = (torch.randn(4, 3, dtype=torch.float64) * 1e4).requires_grad_()

    module(x).square().mean().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert all(parameter.grad is not None for parameter in module.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in module.parameters())


def test_stable_rational_clip_matches_formula_and_preserves_positive_derivative() -> None:
    x = torch.tensor(
        [-1e30, -625.0, -25.1, -25.0, -24.9, 0.0, 24.9, 25.0, 25.1, 625.0, 1e30],
        dtype=torch.float32,
        requires_grad=True,
    )

    actual = _rational_clip(x, 25.0)
    expected = x.detach().double() * torch.rsqrt(1.0 + (x.detach().double() / 25.0).square())
    actual.sum().backward()

    torch.testing.assert_close(actual.double(), expected, rtol=1e-6, atol=1e-6)
    assert torch.all(actual.abs() <= 25.0)
    assert x.grad is not None
    assert torch.all(x.grad >= 0)
    expected_grad = (1.0 + (x.detach().double() / 25.0).square()).pow(-1.5)
    torch.testing.assert_close(
        x.grad[1:-1].double(),
        expected_grad[1:-1],
        rtol=1e-5,
        atol=1e-7,
    )


@pytest.mark.parametrize(
    ("control", "gate_value", "up_value"),
    [
        (SwiGLUChannelControl.situ, 14.0, 87.5),
        (SwiGLUChannelControl.asymmetric_rational_clip, 100.0, 338.0),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_channel_control_uses_fp32_for_low_precision_projections(
    control: SwiGLUChannelControl,
    gate_value: float,
    up_value: float,
    dtype: torch.dtype,
) -> None:
    module = ChannelControlledFeedForward(
        d_model=1,
        hidden_size=1,
        bias=True,
        dtype=dtype,
        control=control,
    )
    with torch.no_grad():
        module.w1.weight.zero_()
        module.w1.bias.fill_(gate_value)
        module.w3.weight.zero_()
        module.w3.bias.fill_(up_value)
        module.w2.weight.fill_(1.0)
        module.w2.bias.zero_()
    x = torch.zeros(1, 1, dtype=dtype, requires_grad=True)

    output = module(x)
    output.sum().backward()

    assert output.dtype == dtype
    assert output.item() <= 100.0
    assert module.w1.bias.grad is not None
    assert module.w3.bias.grad is not None
    assert module.w1.bias.grad.item() > 0
    assert module.w3.bias.grad.item() > 0


def test_negative_gate_overflow_uses_zero_limit() -> None:
    module = ChannelControlledFeedForward(
        d_model=1,
        hidden_size=1,
        bias=True,
        dtype=torch.float16,
        control=SwiGLUChannelControl.asymmetric_rational_clip,
    )
    with torch.no_grad():
        module.w1.weight.fill_(2.0)
        module.w1.bias.zero_()
        module.w3.weight.zero_()
        module.w3.bias.fill_(1.0)
        module.w2.weight.fill_(1.0)
        module.w2.bias.zero_()
    x = torch.tensor([[-40_000.0]], dtype=torch.float16, requires_grad=True)

    output = module(x)
    output.sum().backward()

    assert output.item() == 0.0
    assert torch.isfinite(output).all()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert all(parameter.grad is not None for parameter in module.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in module.parameters())


def test_asymmetric_rational_clip_leaves_negative_gate_linear_factor_unclipped() -> None:
    module = ChannelControlledFeedForward(
        d_model=1,
        hidden_size=1,
        bias=True,
        dtype=torch.float64,
        control=SwiGLUChannelControl.asymmetric_rational_clip,
    )
    with torch.no_grad():
        module.w1.weight.zero_()
        module.w1.bias.fill_(-8.0)
        module.w3.weight.zero_()
        module.w3.bias.fill_(1.0)
        module.w2.weight.fill_(1.0)
        module.w2.bias.zero_()

    output = module(torch.zeros(1, 1, dtype=torch.float64))
    gate = torch.tensor(-8.0, dtype=torch.float64)
    controlled_up = _rational_clip(torch.tensor(1.0, dtype=torch.float64), 25.0)
    expected = gate * torch.sigmoid(gate) * controlled_up
    symmetrically_clipped = (
        _rational_clip(gate, 4.0)
        * torch.sigmoid(gate)
        * controlled_up
    )

    torch.testing.assert_close(output.squeeze(), expected)
    assert not torch.isclose(output.squeeze(), symmetrically_clipped)


def test_fixed_scalars_do_not_change_checkpoint_keys_or_parameter_count() -> None:
    native = FeedForward(d_model=4, hidden_size=8, bias=False)
    controlled = ChannelControlledFeedForward(
        d_model=4,
        hidden_size=8,
        bias=False,
        control=SwiGLUChannelControl.situ,
    )

    assert controlled.beta_gate == 4.0
    assert controlled.beta_up == 25.0
    assert controlled.state_dict().keys() == native.state_dict().keys()
    assert sum(p.numel() for p in controlled.parameters()) == sum(
        p.numel() for p in native.parameters()
    )


def test_fixed_scalars_are_not_configurable() -> None:
    with pytest.raises(TypeError):
        ChannelControlledFeedForwardConfig(hidden_size=8, beta_gate=8.0)
