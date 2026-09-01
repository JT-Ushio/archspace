import sys
from argparse import Namespace
from pathlib import Path

import pytest

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.attention import AttentionConfig
from olmo_core.script_utils import ExperimentConfig
from olmo_ahn import LinearAttentionConfig, LinearAttentionType, SerializableMuonConfig

CFGS_DIR = Path(__file__).parents[1] / "cfgs"
sys.path.insert(0, str(CFGS_DIR))

from _models import (  # noqa: E402
    build_all_linear_olmo3_1b,
    build_hybrid_linear_olmo3_1b,
)
from _pretrain_common import N_BATCH_PER_GPU, build_pretrain_config  # noqa: E402


def _opts(tmp_path: Path, *, sequence_length: int | None = None) -> Namespace:
    return Namespace(
        sequence_length=sequence_length,
        data_root=str(tmp_path / "data"),
        work_dir=str(tmp_path / "work"),
        save_folder=str(tmp_path / "checkpoints"),
        name="config-test",
    )


def test_all_linear_experiment_round_trips_kda_override(tmp_path: Path) -> None:
    config = build_pretrain_config(
        _opts(tmp_path),
        ["model.block.sequence_mixer.attention_type=kda"],
        build_all_linear_olmo3_1b,
        variant="all-linear",
    )

    restored = ExperimentConfig.from_dict(config.as_config_dict())

    assert isinstance(restored.model.block.sequence_mixer, LinearAttentionConfig)
    assert restored.model.block.sequence_mixer.attention_type == LinearAttentionType.kda
    assert isinstance(restored.train_module.optim, SerializableMuonConfig)
    assert restored.train_module.optim.flatten is True
    assert restored.trainer.callbacks["checkpointer"].save_interval == 2500


def test_hybrid_experiment_round_trips_gdn2_override(tmp_path: Path) -> None:
    config = build_pretrain_config(
        _opts(tmp_path),
        ["model.block.sequence_mixer.attention_type=gdn2"],
        build_hybrid_linear_olmo3_1b,
        variant="hybrid-linear",
    )

    restored = ExperimentConfig.from_dict(config.as_config_dict())

    for index, block in enumerate(restored.model.resolved_block_configs):
        if index in (3, 7, 11, 15):
            assert isinstance(block.sequence_mixer, AttentionConfig)
        else:
            assert isinstance(block.sequence_mixer, LinearAttentionConfig)
            assert block.sequence_mixer.attention_type == LinearAttentionType.gdn2


def test_sequence_length_updates_rank_microbatch_size(tmp_path: Path) -> None:
    config = build_pretrain_config(
        _opts(tmp_path, sequence_length=8192),
        [],
        build_all_linear_olmo3_1b,
        variant="all-linear",
    )

    assert config.train_module.rank_microbatch_size == N_BATCH_PER_GPU * 8192


@pytest.mark.parametrize("sequence_length", [0, 3072, 12_288])
def test_invalid_sequence_length_fails_during_config_build(
    tmp_path: Path,
    sequence_length: int,
) -> None:
    with pytest.raises(OLMoConfigurationError, match="sequence_length"):
        build_pretrain_config(
            _opts(tmp_path, sequence_length=sequence_length),
            [],
            build_all_linear_olmo3_1b,
            variant="all-linear",
        )
