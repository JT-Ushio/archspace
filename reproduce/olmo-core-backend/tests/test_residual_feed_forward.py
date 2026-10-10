"""Native behavior and fail-closed shape checks for the new residual FFN."""

import pytest
import torch

from olmo_core.exceptions import OLMoConfigurationError
from olmo_mose.feed_forward import MoSESwiGLUConfig
from olmo_mose.residual_feed_forward import ResidualMoSESwiGLUConfig


def config(**overrides):
    return ResidualMoSESwiGLUConfig(
        hidden_size=overrides.pop("hidden_size", 16),
        r2=8, down_r2=8, **overrides,
    )


def test_disabled_residuals_preserve_existing_ffn_and_gradients():
    cfg = config()
    native = cfg.build(16)
    plain = MoSESwiGLUConfig(
        **{k: v for k, v in cfg.as_dict(exclude_none=True).items()
           if k not in ("wide_residual", "latent_residual")}
    ).build(16)
    plain.load_state_dict(native.state_dict(), strict=True)
    a = torch.randn(2, 3, 16, requires_grad=True)
    b = a.detach().clone().requires_grad_(True)
    left, right = native(a), plain(b)
    torch.testing.assert_close(left, right, atol=0, rtol=0)
    left.square().sum().backward()
    right.square().sum().backward()
    torch.testing.assert_close(a.grad, b.grad, atol=0, rtol=0)
    for (name, p), (other_name, q) in zip(native.named_parameters(), plain.named_parameters()):
        assert name == other_name
        torch.testing.assert_close(p.grad, q.grad, atol=0, rtol=0)


def test_latent_skip_bypasses_zero_down_encoder():
    off = config(latent_residual=False).build(16)
    on = config(latent_residual=True).build(16)
    on.load_state_dict(off.state_dict())
    with torch.no_grad():
        off.down_nonlinear_u.weight.zero_()
        on.down_nonlinear_u.weight.zero_()
    x = torch.randn(2, 3, 16, requires_grad=True)
    assert torch.count_nonzero(off(x)) == 0
    out = on(x)
    assert torch.count_nonzero(out) > 0
    out.square().sum().backward()
    assert torch.isfinite(x.grad).all() and torch.count_nonzero(x.grad) > 0


def test_wide_skip_bypasses_zero_down_readout_without_returning_input():
    off = config(wide_residual=False).build(16)
    on = config(wide_residual=True).build(16)
    on.load_state_dict(off.state_dict())
    with torch.no_grad():
        off.down_nonlinear_v.weight.zero_()
        on.down_nonlinear_v.weight.zero_()
    x = torch.randn(2, 3, 16)
    assert torch.count_nonzero(off(x)) == 0
    hidden = torch.nn.functional.silu(on.gate_nonlinear_v(on.gate_nonlinear_u(x))) * (
        on.up_nonlinear_v(on.up_nonlinear_u(x))
    )
    torch.testing.assert_close(on(x), hidden, atol=0, rtol=0)


def test_invalid_dimensions_and_sharing_fail_explicitly():
    with pytest.raises(OLMoConfigurationError, match="hidden_size"):
        config(hidden_size=64, wide_residual=True).build(16)
    with pytest.raises(OLMoConfigurationError, match="independent UG"):
        config(share_gate_up_subspace=True)
    with pytest.raises(OLMoConfigurationError, match="latent sizes"):
        ResidualMoSESwiGLUConfig(hidden_size=16, r2=8, down_r2=4)


@pytest.mark.parametrize("wide,latent", [(False, False), (True, False), (False, True), (True, True)])
def test_skips_add_no_parameters_or_state(wide, latent):
    module = config(wide_residual=wide, latent_residual=latent).build(16, init_device="meta")
    assert sum(p.numel() for p in module.parameters()) == 6 * 16 * 8
    assert set(module.state_dict()) == set(config().build(16, init_device="meta").state_dict())
