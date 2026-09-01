import pytest

from olmo_core.config import DType
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.attention.recurrent import GatedDeltaNetConfig
from olmo_ahn import LinearAttentionConfig, LinearAttentionType


def test_gdn_parameter_count_delegates_to_native_olmo_config() -> None:
    config = LinearAttentionConfig(attention_type=LinearAttentionType.gdn)
    expected = GatedDeltaNetConfig(
        n_heads=16,
        n_v_heads=16,
        head_dim=128,
        expand_v=2.0,
        allow_neg_eigval=False,
        conv_size=4,
        conv_bias=False,
        norm_eps=1e-5,
        dtype=DType.float32,
    )

    assert config.num_params(2048) == expected.num_params(2048)


@pytest.mark.parametrize("attention_type", [LinearAttentionType.kda, LinearAttentionType.gdn2])
def test_fla_linear_attention_parameter_counts_are_positive(
    attention_type: LinearAttentionType,
) -> None:
    config = LinearAttentionConfig(attention_type=attention_type)

    assert config.num_params(2048) > 0
    assert config._resolved_dimensions(2048) == (16, 128, 128, 1.0)


def test_default_expand_v_tracks_the_selected_algorithm() -> None:
    gdn = LinearAttentionConfig(attention_type=LinearAttentionType.gdn)
    kda = LinearAttentionConfig(attention_type=LinearAttentionType.kda)

    assert gdn._resolved_dimensions(2048)[-1] == 2.0
    assert kda._resolved_dimensions(2048)[-1] == 1.0


@pytest.mark.parametrize("attention_type", list(LinearAttentionType))
def test_qwen35_linear_attention_dimensions_are_supported(
    attention_type: LinearAttentionType,
) -> None:
    config = LinearAttentionConfig(
        attention_type=attention_type,
        conv_size=4,
        head_dim=128,
        n_heads=16,
        n_v_heads=32,
        expand_v=1.0,
    )

    assert config._resolved_dimensions(2048) == (32, 128, 128, 1.0)
    assert config.num_params(2048) > 0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_heads": 0}, "n_heads"),
        ({"n_heads": 8, "n_v_heads": 12}, "n_v_heads"),
        ({"head_dim": 0}, "head_dim"),
        ({"expand_v": 0}, "expand_v"),
        ({"conv_size": 0}, "conv_size"),
        ({"use_short_conv": False}, "use_short_conv"),
        ({"attention_type": "gdn2", "safe_gate": True}, "safe_gate"),
        ({"attention_type": "kda", "safe_gate": True}, "requires lower_bound"),
        ({"attention_type": "kda", "lower_bound": -5.0}, "requires safe_gate"),
        (
            {"attention_type": "kda", "safe_gate": True, "lower_bound": 0.0},
            "safe range",
        ),
        (
            {"attention_type": "kda", "safe_gate": True, "lower_bound": -5.1},
            "safe range",
        ),
    ],
)
def test_invalid_linear_attention_options_fail_early(kwargs: dict, message: str) -> None:
    with pytest.raises(OLMoConfigurationError, match=message):
        LinearAttentionConfig(**kwargs)


def test_explicit_dimensions_must_be_integral() -> None:
    config = LinearAttentionConfig(head_dim=3, expand_v=1.5)

    with pytest.raises(OLMoConfigurationError, match="integer"):
        config.num_params(16)
