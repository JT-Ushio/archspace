import pytest
import torch
import torch.nn.functional as F

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.attention import Attention, AttentionBackendName, AttentionConfig
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.nn.transformer.init import InitMethod
from olmo_mose import (
    LowRankAttention,
    LowRankAttentionConfig,
    NonlinearLowRankProjection,
    patch_low_rank_attention,
)


def test_nonlinear_low_rank_projection_is_exact_u_sigma_v_without_dense_bypass() -> None:
    torch.manual_seed(1)
    projection = NonlinearLowRankProjection(
        in_features=5,
        out_features=7,
        rank=3,
        bias=True,
        dtype=torch.float64,
    )
    x = torch.randn(2, 4, 5, dtype=torch.float64)

    expected = F.linear(
        F.silu(F.linear(x, projection.v.weight)),
        projection.u.weight,
        projection.u.bias,
    )
    torch.testing.assert_close(projection(x), expected)
    assert set(projection.state_dict()) == {"v.weight", "u.weight", "u.bias"}


def test_low_rank_attention_config_parameter_count_matches_module() -> None:
    config = LowRankAttentionConfig(
        n_heads=4,
        n_kv_heads=2,
        head_dim=2,
        rank=3,
        bias=True,
        backend=AttentionBackendName.torch,
    )
    module = config.build(d_model=8, layer_idx=0, n_layers=1)

    assert isinstance(module, LowRankAttention)
    assert config.num_params(8) == sum(parameter.numel() for parameter in module.parameters())


def test_low_rank_attention_uses_four_independent_projections_and_runs_forward() -> None:
    config = LowRankAttentionConfig(
        n_heads=4,
        n_kv_heads=2,
        head_dim=2,
        rank=3,
        bias=False,
        backend=AttentionBackendName.torch,
    )
    module = config.build(d_model=8, layer_idx=0, n_layers=1)
    module.init_weights(
        init_method=InitMethod.normal,
        d_model=8,
        block_idx=0,
        num_blocks=1,
    )
    projections = (module.w_q, module.w_k, module.w_v, module.w_out)

    assert all(isinstance(projection, NonlinearLowRankProjection) for projection in projections)
    assert len({id(projection.v) for projection in projections}) == 4
    assert len({id(projection.u) for projection in projections}) == 4
    dense_keys = ("w_q.weight", "w_k.weight", "w_v.weight", "w_out.weight")
    assert not any(key in module.state_dict() for key in dense_keys)

    x = torch.randn(2, 5, 8, requires_grad=True)
    output = module(x)
    output.sum().backward()

    assert output.shape == x.shape
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(parameter.grad is not None for parameter in module.parameters())


def test_low_rank_attention_supports_independent_rms_norm_activations() -> None:
    config = LowRankAttentionConfig(
        n_heads=2,
        n_kv_heads=2,
        head_dim=4,
        rank=3,
        q_nonlinearity="rms_norm",
        k_nonlinearity="silu",
        v_nonlinearity="rms_norm",
        o_nonlinearity="silu",
        backend=AttentionBackendName.torch,
    )
    module = config.build(d_model=8, layer_idx=0, n_layers=1)
    assert module.w_q.nonlinear_norm is not None
    assert module.w_k.nonlinear_norm is None
    assert module.w_v.nonlinear_norm is not None
    assert module.w_out.nonlinear_norm is None
    assert config.num_params(8) == sum(parameter.numel() for parameter in module.parameters())


def test_disabled_attention_patch_preserves_native_config_exactly() -> None:
    original = TransformerConfig.olmo3_1M(
        vocab_size=128,
        attn_backend=AttentionBackendName.torch,
    )
    patched = patch_low_rank_attention(original, enabled=False, rank=4)

    assert patched.as_config_dict() == original.as_config_dict()
    assert type(patched.block.sequence_mixer) is AttentionConfig
    assert isinstance(patched.build(init_device="meta").blocks["0"].attention, Attention)


def test_enabled_attention_patch_builds_and_initializes_every_layer() -> None:
    config = patch_low_rank_attention(
        TransformerConfig.olmo3_1M(
            vocab_size=128,
            attn_backend=AttentionBackendName.torch,
        ),
        enabled=True,
        rank=4,
    )

    assert all(
        isinstance(block.sequence_mixer, LowRankAttentionConfig)
        and block.sequence_mixer.rank == 4
        for block in config.resolved_block_configs
    )

    model = config.build(init_device="meta")
    model.init_weights(device=torch.device("cpu"))
    for block in model.blocks.values():
        assert isinstance(block.attention, LowRankAttention)
        assert all(
            projection.weight.count_nonzero() > 0
            for projection in block.attention.projection_modules()
        )


@pytest.mark.parametrize("rank", [0, -1, True])
def test_low_rank_attention_rejects_invalid_rank(rank: int) -> None:
    with pytest.raises(OLMoConfigurationError, match="positive integer"):
        LowRankAttentionConfig(rank=rank)


def test_low_rank_attention_rejects_tensor_parallelism() -> None:
    module = LowRankAttentionConfig(
        n_heads=2,
        rank=4,
        backend=AttentionBackendName.torch,
    ).build(d_model=8, layer_idx=0, n_layers=1)

    with pytest.raises(NotImplementedError, match="tensor parallelism"):
        module.apply_tp(None)  # type: ignore[arg-type]
