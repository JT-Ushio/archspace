import pytest
import torch
import torch.nn.functional as F

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.feed_forward import FeedForward
from olmo_mose import (
    ChannelControlledFeedForward,
    ChannelControlledFeedForwardConfig,
    SwiGLUChannelControl,
    SwiGLUChannelControlScope,
)
from olmo_mose.feed_forward import _apply_swiglu_channel_control, _rational_clip


def _reference_hidden(
    gate: torch.Tensor,
    up: torch.Tensor,
    control: SwiGLUChannelControl,
    control_scope: SwiGLUChannelControlScope,
    beta_gate: float,
    beta_up: float,
    rms_beta_scale: float = 4.0,
) -> torch.Tensor:
    if (
        control == SwiGLUChannelControl.standard
        or control_scope == SwiGLUChannelControlScope.none
    ):
        return F.silu(gate) * up
    control_gate = control_scope in (
        SwiGLUChannelControlScope.both,
        SwiGLUChannelControlScope.gate,
    )
    control_up = control_scope in (
        SwiGLUChannelControlScope.both,
        SwiGLUChannelControlScope.up,
    )
    gate_factor = F.silu(gate)
    if control in (
        SwiGLUChannelControl.situ,
        SwiGLUChannelControl.situ_rms,
    ):
        if control_gate:
            gate_linear = beta_gate * torch.tanh(gate / beta_gate)
            gate_factor = gate_linear * torch.sigmoid(gate)
        if control_up:
            if control == SwiGLUChannelControl.situ:
                up = beta_up * torch.tanh(up / beta_up)
            else:
                rms = up.square().mean(dim=-1, keepdim=True)
                rms = rms.clamp_min(torch.finfo(up.dtype).tiny).sqrt()
                beta = rms_beta_scale * rms
                up = beta * torch.tanh(up / beta)
    elif control == SwiGLUChannelControl.asymmetric_rational_clip:
        if control_gate:
            gate_linear = gate * torch.rsqrt(
                1.0 + (F.relu(gate) / beta_gate).square()
            )
            gate_factor = gate_linear * torch.sigmoid(gate)
        if control_up:
            up = up * torch.rsqrt(1.0 + (up / beta_up).square())
    elif control in (
        SwiGLUChannelControl.dpskv4_clip,
        SwiGLUChannelControl.dpskv4_clip_situ,
    ):
        if control_gate:
            gate = gate.clamp_max(10.0)
        gate_factor = F.silu(gate)
        if control_up:
            if control == SwiGLUChannelControl.dpskv4_clip:
                up = up.clamp(min=-10.0, max=10.0)
            else:
                up = beta_up * torch.tanh(up / beta_up)
    else:
        raise NotImplementedError(control)
    return gate_factor * up


@pytest.mark.parametrize("control", list(SwiGLUChannelControl))
@pytest.mark.parametrize("control_scope", list(SwiGLUChannelControlScope))
def test_forward_matches_reference(
    control: SwiGLUChannelControl,
    control_scope: SwiGLUChannelControlScope,
) -> None:
    torch.manual_seed(0)
    module = ChannelControlledFeedForwardConfig(
        hidden_size=7,
        bias=True,
        control=control,
        control_scope=control_scope,
    ).build(d_model=5, dtype=torch.float64)
    x = torch.randn(2, 3, 5, dtype=torch.float64)

    gate = module.w1(x)
    up = module.w3(x)
    expected = module.w2(
        _reference_hidden(
            gate,
            up,
            control,
            control_scope,
            module.beta_gate,
            module.beta_up,
            module.rms_beta_scale,
        )
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
    [
        SwiGLUChannelControl.situ,
        SwiGLUChannelControl.situ_rms,
        SwiGLUChannelControl.asymmetric_rational_clip,
        SwiGLUChannelControl.dpskv4_clip,
        SwiGLUChannelControl.dpskv4_clip_situ,
    ],
)
def test_none_scope_matches_native_swiglu(control: SwiGLUChannelControl) -> None:
    torch.manual_seed(2)
    native = FeedForward(d_model=4, hidden_size=6, bias=False, dtype=torch.float64)
    controlled = ChannelControlledFeedForward(
        d_model=4,
        hidden_size=6,
        bias=False,
        dtype=torch.float64,
        control=control,
        control_scope=SwiGLUChannelControlScope.none,
    )
    controlled.load_state_dict(native.state_dict(), strict=True)
    x = torch.randn(2, 4, dtype=torch.float64)

    torch.testing.assert_close(controlled(x), native(x))


@pytest.mark.parametrize(
    "control",
    [
        SwiGLUChannelControl.situ,
        SwiGLUChannelControl.situ_rms,
        SwiGLUChannelControl.asymmetric_rational_clip,
        SwiGLUChannelControl.dpskv4_clip,
        SwiGLUChannelControl.dpskv4_clip_situ,
    ],
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


@pytest.mark.parametrize(
    ("control", "expected_up"),
    [
        (
            SwiGLUChannelControl.dpskv4_clip,
            torch.tensor([-10.0, 5.0, 10.0], dtype=torch.float64),
        ),
        (
            SwiGLUChannelControl.dpskv4_clip_situ,
            25.0
            * torch.tanh(
                torch.tensor([-20.0, 5.0, 20.0], dtype=torch.float64) / 25.0
            ),
        ),
    ],
)
def test_dpskv4_controls_match_exact_both_scope_formula(
    control: SwiGLUChannelControl,
    expected_up: torch.Tensor,
) -> None:
    gate = torch.tensor([-20.0, 5.0, 20.0], dtype=torch.float64)
    up = torch.tensor([-20.0, 5.0, 20.0], dtype=torch.float64)

    actual = _apply_swiglu_channel_control(
        gate,
        up,
        control=control,
        control_scope=SwiGLUChannelControlScope.both,
        beta_gate=4.0,
        beta_up=25.0,
    )
    expected = F.silu(gate.clamp_max(10.0)) * expected_up.to(torch.float64)

    torch.testing.assert_close(actual, expected)


def test_situ_rms_up_matches_per_token_formula_and_backpropagates_through_rms() -> None:
    gate = torch.tensor(
        [[-2.0, 0.5, 3.0], [1.0, -1.0, 2.0]],
        dtype=torch.float64,
    )
    up = torch.tensor(
        [[-8.0, 2.0, 6.0], [1.0, -4.0, 12.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    reference_up = up.detach().clone().requires_grad_()

    actual = _apply_swiglu_channel_control(
        gate,
        up,
        control=SwiGLUChannelControl.situ_rms,
        control_scope=SwiGLUChannelControlScope.up,
        beta_gate=4.0,
        beta_up=25.0,
        rms_beta_scale=4.0,
    )
    rms = reference_up.square().mean(dim=-1, keepdim=True).sqrt()
    beta = 4.0 * rms
    expected = F.silu(gate) * beta * torch.tanh(reference_up / beta)

    actual.sum().backward()
    expected.sum().backward()

    torch.testing.assert_close(actual, expected)
    assert up.grad is not None and reference_up.grad is not None
    torch.testing.assert_close(up.grad, reference_up.grad)


def test_situ_rms_up_is_finite_for_an_all_zero_token() -> None:
    gate = torch.ones(1, 3, dtype=torch.float64, requires_grad=True)
    up = torch.zeros(1, 3, dtype=torch.float64, requires_grad=True)

    output = _apply_swiglu_channel_control(
        gate,
        up,
        control=SwiGLUChannelControl.situ_rms,
        control_scope=SwiGLUChannelControlScope.up,
        beta_gate=4.0,
        beta_up=25.0,
        rms_beta_scale=4.0,
    )
    output.sum().backward()

    assert torch.equal(output, torch.zeros_like(output))
    assert gate.grad is not None and torch.isfinite(gate.grad).all()
    assert up.grad is not None and torch.isfinite(up.grad).all()


@pytest.mark.parametrize(
    "control",
    [
        SwiGLUChannelControl.dpskv4_clip,
        SwiGLUChannelControl.dpskv4_clip_situ,
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_dpskv4_controls_use_fp32_for_low_precision_projections(
    control: SwiGLUChannelControl,
    dtype: torch.dtype,
) -> None:
    gate = torch.tensor([20.0], dtype=dtype, requires_grad=True)
    up = torch.tensor([20.0], dtype=dtype, requires_grad=True)

    actual = _apply_swiglu_channel_control(
        gate,
        up,
        control=control,
        control_scope=SwiGLUChannelControlScope.both,
        beta_gate=4.0,
        beta_up=25.0,
    )
    controlled_up = (
        up.float().clamp(min=-10.0, max=10.0)
        if control == SwiGLUChannelControl.dpskv4_clip
        else 25.0 * torch.tanh(up.float() / 25.0)
    )
    expected = (F.silu(gate.float().clamp_max(10.0)) * controlled_up).to(dtype)
    actual.sum().backward()

    assert actual.dtype == dtype
    torch.testing.assert_close(actual, expected)
    assert gate.grad is not None and torch.isfinite(gate.grad).all()
    assert up.grad is not None and torch.isfinite(up.grad).all()


@pytest.mark.parametrize(
    "control",
    [
        SwiGLUChannelControl.situ,
        SwiGLUChannelControl.situ_rms,
        SwiGLUChannelControl.asymmetric_rational_clip,
        SwiGLUChannelControl.dpskv4_clip,
        SwiGLUChannelControl.dpskv4_clip_situ,
    ],
)
def test_fixed_controls_do_not_change_checkpoint_keys_or_parameter_count(
    control: SwiGLUChannelControl,
) -> None:
    native = FeedForward(d_model=4, hidden_size=8, bias=False)
    controlled = ChannelControlledFeedForward(
        d_model=4,
        hidden_size=8,
        bias=False,
        control=control,
    )

    assert controlled.beta_gate == 4.0
    assert controlled.beta_up == 25.0
    assert controlled.rms_beta_scale == 4.0
    assert controlled.state_dict().keys() == native.state_dict().keys()
    assert sum(p.numel() for p in controlled.parameters()) == sum(
        p.numel() for p in native.parameters()
    )


def test_fixed_scalars_are_not_configurable() -> None:
    with pytest.raises(TypeError):
        ChannelControlledFeedForwardConfig(hidden_size=8, beta_gate=8.0)


@pytest.mark.parametrize("scale", [0.0, -1.0, float("inf"), float("nan"), True])
def test_situ_rms_rejects_invalid_beta_scale(scale: float) -> None:
    with pytest.raises(OLMoConfigurationError, match="rms_beta_scale"):
        ChannelControlledFeedForwardConfig(hidden_size=8, rms_beta_scale=scale)
