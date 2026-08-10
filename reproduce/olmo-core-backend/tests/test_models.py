import sys
from pathlib import Path

import pytest
import torch

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.feed_forward import FeedForwardConfig
from olmo_core.nn.transformer import TransformerConfig
from olmo_mose import (
    ChannelControlledFeedForwardConfig,
    MoSENonlinearity,
    MoSESwiGLU,
    MoSESwiGLUConfig,
    SerializableMuonConfig,
    SwiGLUChannelControl,
    patch_mose_swiglu,
    patch_swiglu_channel_control,
)

CFGS_DIR = Path(__file__).parents[1] / "cfgs"
sys.path.insert(0, str(CFGS_DIR))

from _models import build_mose_olmo3_1b, build_olmo3_1b  # noqa: E402


class _Tokenizer:
    def padded_vocab_size(self) -> int:
        return 100_352


def test_standard_model_is_an_unchanged_olmo3_1b_config() -> None:
    expected = TransformerConfig.olmo3_1B(vocab_size=100_352)
    actual = build_olmo3_1b(_Tokenizer(), SwiGLUChannelControl.standard)

    assert actual.as_config_dict() == expected.as_config_dict()
    assert type(actual.block.feed_forward) is FeedForwardConfig


def test_channel_control_patches_olmo3_1b_without_changing_parameter_count() -> None:
    baseline = build_olmo3_1b(_Tokenizer(), SwiGLUChannelControl.standard)
    controlled = build_olmo3_1b(
        _Tokenizer(), SwiGLUChannelControl.asymmetric_rational_clip
    )
    feed_forward = controlled.block.feed_forward

    assert isinstance(feed_forward, ChannelControlledFeedForwardConfig)
    assert feed_forward.control == SwiGLUChannelControl.asymmetric_rational_clip
    assert controlled.num_params == baseline.num_params
    assert type(baseline.block.feed_forward) is FeedForwardConfig


def test_custom_feed_forward_survives_control_override_round_trip() -> None:
    config = build_olmo3_1b(_Tokenizer(), SwiGLUChannelControl.situ)

    overridden = config.merge(
        [
            "block.feed_forward.control=asymmetric_rational_clip",
        ]
    )
    feed_forward = overridden.block.feed_forward

    assert isinstance(feed_forward, ChannelControlledFeedForwardConfig)
    assert feed_forward.control == SwiGLUChannelControl.asymmetric_rational_clip


def test_standard_patch_restores_native_feed_forward_config() -> None:
    controlled = build_olmo3_1b(_Tokenizer(), SwiGLUChannelControl.situ)

    restored = patch_swiglu_channel_control(
        controlled,
        control=SwiGLUChannelControl.standard,
    )

    assert type(restored.block.feed_forward) is FeedForwardConfig
    assert isinstance(controlled.block.feed_forward, ChannelControlledFeedForwardConfig)


def test_controlled_feed_forward_builds_inside_transformer() -> None:
    config = TransformerConfig.olmo3_1M(
        vocab_size=128,
        attn_backend=AttentionBackendName.torch,
    )
    config.block.feed_forward = ChannelControlledFeedForwardConfig(
        hidden_size=config.block.feed_forward.hidden_size,
        bias=config.block.feed_forward.bias,
        dtype=config.block.feed_forward.dtype,
        control=SwiGLUChannelControl.situ,
    )

    model = config.build(init_device="meta")

    assert model.blocks["0"].feed_forward.control == SwiGLUChannelControl.situ


def test_mose_model_uses_configurable_default_ranks() -> None:
    baseline = build_olmo3_1b(_Tokenizer(), SwiGLUChannelControl.standard)
    config = build_mose_olmo3_1b(_Tokenizer(), SwiGLUChannelControl.situ)
    feed_forward = config.block.feed_forward

    assert isinstance(feed_forward, MoSESwiGLUConfig)
    assert (feed_forward.r1, feed_forward.r2) == (880, 880)
    assert (feed_forward.down_r1, feed_forward.down_r2) == (880, 880)
    assert config.num_params == 1_487_013_888
    assert config.num_params - baseline.num_params == 2_097_152


def test_mose_rank_and_control_overrides_round_trip() -> None:
    config = build_mose_olmo3_1b(_Tokenizer(), SwiGLUChannelControl.situ)

    overridden = config.merge(
        [
            "block.feed_forward.r1=64",
            "block.feed_forward.r2=32",
            "block.feed_forward.down_r1=16",
            "block.feed_forward.down_r2=0",
            "block.feed_forward.control=asymmetric_rational_clip",
            "block.feed_forward.gate_nonlinearity=rms_norm",
            "block.feed_forward.up_nonlinearity=silu",
            "block.feed_forward.down_nonlinearity=rms_norm",
        ]
    )
    feed_forward = overridden.block.feed_forward

    assert isinstance(feed_forward, MoSESwiGLUConfig)
    assert (feed_forward.r1, feed_forward.r2) == (64, 32)
    assert (feed_forward.down_r1, feed_forward.down_r2) == (16, 0)
    assert feed_forward.control == SwiGLUChannelControl.asymmetric_rational_clip
    assert feed_forward.gate_nonlinearity == MoSENonlinearity.rms_norm
    assert feed_forward.up_nonlinearity == MoSENonlinearity.silu
    assert feed_forward.down_nonlinearity == MoSENonlinearity.rms_norm


def test_mose_rmsnorm_silu_baseline_reaches_model_config() -> None:
    config = build_mose_olmo3_1b(
        _Tokenizer(),
        SwiGLUChannelControl.standard,
        gate_nonlinearity=MoSENonlinearity.rms_norm,
        up_nonlinearity=MoSENonlinearity.rms_norm,
        down_nonlinearity=MoSENonlinearity.silu,
    )
    feed_forward = config.block.feed_forward

    assert isinstance(feed_forward, MoSESwiGLUConfig)
    assert feed_forward.control == SwiGLUChannelControl.standard
    assert feed_forward.gate_nonlinearity == MoSENonlinearity.rms_norm
    assert feed_forward.up_nonlinearity == MoSENonlinearity.rms_norm
    assert feed_forward.down_nonlinearity == MoSENonlinearity.silu


def test_native_control_patch_rejects_mose_topology() -> None:
    config = build_mose_olmo3_1b(_Tokenizer(), SwiGLUChannelControl.situ)

    with pytest.raises(OLMoConfigurationError, match="cannot change MoSE topology"):
        patch_swiglu_channel_control(config, control=SwiGLUChannelControl.standard)


def _tiny_mose_config() -> TransformerConfig:
    return patch_mose_swiglu(
        TransformerConfig.olmo3_1M(
            vocab_size=128,
            attn_backend=AttentionBackendName.torch,
        ),
        control=SwiGLUChannelControl.asymmetric_rational_clip,
        r1=4,
        r2=3,
        down_r1=4,
        down_r2=3,
    )


def test_mose_transformer_initializes_all_projections_deterministically() -> None:
    models = []
    for _ in range(2):
        model = _tiny_mose_config().build(init_device="meta")
        model.init_weights(device=torch.device("cpu"))
        models.append(model)

    first, second = models
    for parameter in first.parameters():
        assert not parameter.is_meta
        assert torch.isfinite(parameter).all()
    for key, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[key], rtol=0, atol=0)

    for block in first.blocks.values():
        assert isinstance(block.feed_forward, MoSESwiGLU)
        assert all(
            projection.weight.count_nonzero() > 0
            for projection in block.feed_forward.projection_modules()
        )


def test_muon_categorizes_every_mose_projection_as_a_matrix() -> None:
    model = _tiny_mose_config().build(init_device="meta")

    optim = SerializableMuonConfig()
    categories = optim.categorize_parameters(model)
    mose_weights = {
        name
        for name, parameter in model.named_parameters()
        if ".feed_forward." in name and parameter.ndim == 2
    }

    assert mose_weights
    assert mose_weights <= set(categories["matrix"])

    optim.build_groups(model)
    assert optim.group_overrides is None
    restored = SerializableMuonConfig.from_dict(optim.as_config_dict())
    assert restored.group_overrides is None


def test_mose_patch_preserves_native_unspecified_bias_semantics() -> None:
    config = TransformerConfig.olmo3_1M(
        vocab_size=128,
        attn_backend=AttentionBackendName.torch,
    )
    config.block.feed_forward = FeedForwardConfig(hidden_size=16, bias=None)

    patched = patch_mose_swiglu(
        config,
        control=SwiGLUChannelControl.situ,
        r1=4,
        r2=3,
        down_r1=0,
        down_r2=0,
    )
    feed_forward = patched.block.feed_forward

    assert isinstance(feed_forward, MoSESwiGLUConfig)
    assert feed_forward.bias is True
    assert feed_forward.num_params(patched.d_model) == sum(
        parameter.numel() for parameter in feed_forward.build(patched.d_model).parameters()
    )
