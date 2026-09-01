import sys
from pathlib import Path

import pytest

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.attention import AttentionBackendName, AttentionConfig
from olmo_core.nn.transformer import TransformerConfig
from olmo_ahn import (
    LinearAttentionConfig,
    LinearAttentionType,
    patch_all_linear_attention,
    patch_swa_with_linear_attention,
)

CFGS_DIR = Path(__file__).parents[1] / "cfgs"
sys.path.insert(0, str(CFGS_DIR))

from _models import (  # noqa: E402
    build_all_linear_olmo3_1b,
    build_hybrid_linear_olmo3_1b,
    build_olmo3_1b_baseline,
)


class _Tokenizer:
    def padded_vocab_size(self) -> int:
        return 100_352


def test_baseline_is_the_unmodified_olmo3_1b_config() -> None:
    expected = TransformerConfig.olmo3_1B(
        vocab_size=100_352,
        attn_backend=AttentionBackendName.flash_3,
    )

    assert build_olmo3_1b_baseline(_Tokenizer()).as_config_dict() == expected.as_config_dict()


def test_all_linear_replaces_every_layer_without_mutating_input() -> None:
    baseline = build_olmo3_1b_baseline(_Tokenizer())
    before = baseline.as_config_dict()
    patched = patch_all_linear_attention(baseline)

    assert baseline.as_config_dict() == before
    assert patched.block_overrides is None
    assert all(
        isinstance(block.sequence_mixer, LinearAttentionConfig)
        for block in patched.resolved_block_configs
    )
    assert all(
        block.sequence_mixer.attention_type == LinearAttentionType.gdn
        for block in patched.resolved_block_configs
    )


def test_hybrid_replaces_exactly_the_three_swa_layers_in_each_group() -> None:
    baseline = build_olmo3_1b_baseline(_Tokenizer())
    hybrid = build_hybrid_linear_olmo3_1b(_Tokenizer())
    linear_indices = [
        index
        for index, block in enumerate(hybrid.resolved_block_configs)
        if isinstance(block.sequence_mixer, LinearAttentionConfig)
    ]
    full_indices = [
        index
        for index, block in enumerate(hybrid.resolved_block_configs)
        if isinstance(block.sequence_mixer, AttentionConfig)
    ]

    assert linear_indices == [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]
    assert full_indices == [3, 7, 11, 15]
    for index in full_indices:
        assert (
            hybrid.resolved_block_configs[index].as_config_dict()
            == baseline.resolved_block_configs[index].as_config_dict()
        )


@pytest.mark.parametrize("attention_type", list(LinearAttentionType))
def test_all_linear_builder_supports_every_requested_type(
    attention_type: LinearAttentionType,
) -> None:
    config = build_all_linear_olmo3_1b(_Tokenizer(), attention_type)

    assert all(
        block.sequence_mixer.attention_type == attention_type
        for block in config.resolved_block_configs
    )


@pytest.mark.parametrize("attention_type", list(LinearAttentionType))
def test_hybrid_builder_supports_every_requested_type(
    attention_type: LinearAttentionType,
) -> None:
    config = build_hybrid_linear_olmo3_1b(_Tokenizer(), attention_type)

    for index, block in enumerate(config.resolved_block_configs):
        if index in (3, 7, 11, 15):
            assert isinstance(block.sequence_mixer, AttentionConfig)
        else:
            assert isinstance(block.sequence_mixer, LinearAttentionConfig)
            assert block.sequence_mixer.attention_type == attention_type


def test_cli_style_override_switches_the_all_linear_algorithm() -> None:
    config = build_all_linear_olmo3_1b(_Tokenizer())

    overridden = config.merge(["block.sequence_mixer.attention_type=kda"])

    assert overridden.block.sequence_mixer.attention_type == LinearAttentionType.kda


def test_parameter_count_tracks_the_replaced_layer_count() -> None:
    baseline = build_olmo3_1b_baseline(_Tokenizer())
    all_linear = build_all_linear_olmo3_1b(_Tokenizer())
    hybrid = build_hybrid_linear_olmo3_1b(_Tokenizer())
    baseline_attention = baseline.block.sequence_mixer.num_params(baseline.d_model)
    linear_attention = all_linear.block.sequence_mixer.num_params(all_linear.d_model)
    difference = linear_attention - baseline_attention

    assert all_linear.num_params == baseline.num_params + 16 * difference
    assert hybrid.num_params == baseline.num_params + 12 * difference


def test_patch_rejects_existing_block_overrides() -> None:
    config = build_olmo3_1b_baseline(_Tokenizer())
    config.block_overrides = {0: config.block.copy()}

    with pytest.raises(OLMoConfigurationError, match="block_overrides"):
        patch_swa_with_linear_attention(config)
