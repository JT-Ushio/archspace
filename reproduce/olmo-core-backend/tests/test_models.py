import sys
from pathlib import Path

import pytest
import torch

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.attention import AttentionBackendName, AttentionConfig
from olmo_core.nn.transformer import TransformerConfig
from olmo_cubit import (
    CubitAttention,
    CubitAttentionConfig,
    SerializableMuonConfig,
    patch_cubit,
)

CFGS_DIR = Path(__file__).parents[1] / "cfgs"
sys.path.insert(0, str(CFGS_DIR))

from _models import build_cubit_olmo3_1b, build_olmo3_1b  # noqa: E402


class _Tokenizer:
    def padded_vocab_size(self) -> int:
        return 100_352


def test_baseline_is_olmo3_1b_with_flash_attention_3() -> None:
    expected = TransformerConfig.olmo3_1B(
        vocab_size=100_352,
        attn_backend=AttentionBackendName.flash_3,
    )
    actual = build_olmo3_1b(_Tokenizer())

    assert actual.as_config_dict() == expected.as_config_dict()
    assert type(actual.block.sequence_mixer) is AttentionConfig


def test_cubit_patch_is_non_mutating_and_counts_added_parameters() -> None:
    baseline = build_olmo3_1b(_Tokenizer())
    cubit = build_cubit_olmo3_1b(_Tokenizer())
    attention = cubit.block.sequence_mixer

    assert isinstance(attention, CubitAttentionConfig)
    assert type(baseline.block.sequence_mixer) is AttentionConfig
    assert attention.backend == AttentionBackendName.flash_3

    d_model = cubit.d_model
    n_heads = attention.n_heads
    head_dim = attention.head_dim or d_model // n_heads
    added_per_layer = d_model * n_heads * head_dim + d_model * n_heads + 4 * n_heads
    assert cubit.num_params == baseline.num_params + cubit.n_layers * added_per_layer


def test_shared_reference_omits_the_extra_reference_projection() -> None:
    independent = build_cubit_olmo3_1b(_Tokenizer())
    shared = build_cubit_olmo3_1b(_Tokenizer(), share_reference=True)

    assert independent.num_params - shared.num_params == independent.n_layers * 2048 * 2048


def test_partial_layer_range_is_half_open() -> None:
    config = build_cubit_olmo3_1b(
        _Tokenizer(),
        cubit_start_layer=2,
        cubit_end_layer=15,
    )
    resolved = config.resolved_block_configs

    assert all(type(block.sequence_mixer) is AttentionConfig for block in resolved[:2])
    assert all(isinstance(block.sequence_mixer, CubitAttentionConfig) for block in resolved[2:15])
    assert type(resolved[15].sequence_mixer) is AttentionConfig


def test_cubit_config_round_trips_and_accepts_overrides() -> None:
    config = build_cubit_olmo3_1b(_Tokenizer()).merge(
        [
            "block.sequence_mixer.regularization=0.001",
            "block.sequence_mixer.lrr_lower=0.25",
            "block.sequence_mixer.lrr_upper=1.75",
        ]
    )
    restored = TransformerConfig.from_dict(config.as_config_dict())
    attention = restored.block.sequence_mixer

    assert isinstance(attention, CubitAttentionConfig)
    assert attention.regularization == pytest.approx(0.001)
    assert attention.lrr_lower == pytest.approx(0.25)
    assert attention.lrr_upper == pytest.approx(1.75)


def _tiny_config() -> TransformerConfig:
    return patch_cubit(
        TransformerConfig.olmo3_1M(
            vocab_size=128,
            attn_backend=AttentionBackendName.torch,
        ),
        regularization=1e-4,
    )


def test_tiny_transformer_builds_and_initializes_deterministically() -> None:
    models = []
    for _ in range(2):
        model = _tiny_config().build(init_device="meta")
        model.init_weights(device=torch.device("cpu"))
        models.append(model)

    first, second = models
    for key, value in first.state_dict().items():
        assert not value.is_meta
        assert torch.isfinite(value).all()
        torch.testing.assert_close(value, second.state_dict()[key], rtol=0, atol=0)

    for block in first.blocks.values():
        assert isinstance(block.attention, CubitAttention)
        assert block.attention.w_r is not None
        torch.testing.assert_close(block.attention.reference_scale, torch.ones(4))
        torch.testing.assert_close(block.attention.lrr_lower, torch.full((4,), 0.5))
        torch.testing.assert_close(block.attention.lrr_range, torch.full((4,), 1.5))
        torch.testing.assert_close(
            block.attention.log_regularization,
            torch.full((4,), torch.log(torch.tensor(1e-4)).item()),
        )


def test_tiny_transformer_forward_and_backward_are_finite() -> None:
    model = _tiny_config().build(init_device="meta")
    model.init_weights(device=torch.device("cpu"))
    input_ids = torch.randint(0, 128, (2, 8))

    logits = model(input_ids)
    logits.float().square().mean().backward()

    assert logits.shape == (2, 8, 128)
    assert torch.isfinite(logits).all()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_muon_categorizes_cubit_projections_as_matrices() -> None:
    model = _tiny_config().build(init_device="meta")
    optim = SerializableMuonConfig()
    categories = optim.categorize_parameters(model)
    cubit_matrices = {
        name
        for name, parameter in model.named_parameters()
        if ".attention." in name
        and parameter.ndim == 2
        and name.rsplit(".", 2)[-2] in {"w_r", "w_lrr"}
    }

    assert cubit_matrices
    assert cubit_matrices <= set(categories["matrix"])
    optim.build_groups(model)
    assert optim.group_overrides is None
    restored = SerializableMuonConfig.from_dict(optim.as_config_dict())
    assert restored.group_overrides is None


@pytest.mark.parametrize(
    ("start", "end"),
    [(-1, None), (0, 0), (2, 2), (3, 2), (0, 5), (True, None), (0, True)],
)
def test_patch_rejects_invalid_layer_ranges(start, end) -> None:
    config = TransformerConfig.olmo3_1M(
        vocab_size=128,
        attn_backend=AttentionBackendName.torch,
    )

    with pytest.raises(OLMoConfigurationError, match="layer range"):
        patch_cubit(config, cubit_start_layer=start, cubit_end_layer=end)
