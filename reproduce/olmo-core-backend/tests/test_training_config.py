import sys
from argparse import Namespace
from pathlib import Path

import pytest

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.script_utils import ExperimentConfig
from olmo_cubit import CubitAttentionConfig

CFGS_DIR = Path(__file__).parents[1] / "cfgs"
sys.path.insert(0, str(CFGS_DIR))

from _models import build_cubit_olmo3_1b  # noqa: E402
from _pretrain_common import (  # noqa: E402
    DEFAULT_SEQUENCE_LENGTH,
    RANK_MICROBATCH_SEQUENCES,
    build_pretrain_config,
    get_cubit_cli_parser,
)


def _opts(tmp_path: Path, sequence_length=None) -> Namespace:
    return Namespace(
        sequence_length=sequence_length,
        data_root=str(tmp_path / "data"),
        work_dir=str(tmp_path / "work"),
        save_folder=str(tmp_path / "checkpoints"),
        name="cubit-test",
    )


def test_cubit_cli_parser_accepts_layer_range() -> None:
    opts = get_cubit_cli_parser().parse_args(
        [
            "--save-folder=/tmp/checkpoints",
            "--cubit-start-layer=2",
            "--cubit-end-layer=15",
        ]
    )

    assert opts.cubit_start_layer == 2
    assert opts.cubit_end_layer == 15


def test_experiment_config_round_trips(tmp_path: Path) -> None:
    config = build_pretrain_config(
        _opts(tmp_path),
        [],
        build_cubit_olmo3_1b,
        variant="cubit-fa3",
    )
    restored = ExperimentConfig.from_dict(config.as_config_dict())
    attention = restored.model.block.sequence_mixer

    assert isinstance(attention, CubitAttentionConfig)
    assert attention.backend == AttentionBackendName.flash_3
    assert attention.krr_implementation == "streaming"
    assert attention.krr_block_size == 64
    assert restored.train_module.compile_model is False
    assert restored.trainer.callbacks["checkpointer"].save_interval == 2500
    assert restored.train_module.rank_microbatch_size == (
        RANK_MICROBATCH_SEQUENCES * DEFAULT_SEQUENCE_LENGTH
    )


def test_sequence_length_updates_rank_microbatch_size(tmp_path: Path) -> None:
    config = build_pretrain_config(
        _opts(tmp_path, sequence_length=512),
        [],
        build_cubit_olmo3_1b,
        variant="cubit-fa3",
    )

    assert config.train_module.rank_microbatch_size == RANK_MICROBATCH_SEQUENCES * 512


def test_compile_model_can_be_enabled_by_override(tmp_path: Path) -> None:
    config = build_pretrain_config(
        _opts(tmp_path),
        ["train_module.compile_model=true"],
        build_cubit_olmo3_1b,
        variant="cubit-fa3",
    )

    assert config.train_module.compile_model is True


@pytest.mark.parametrize("sequence_length", [0, 768, 12_288])
def test_invalid_sequence_length_fails_during_config_build(
    tmp_path: Path,
    sequence_length: int,
) -> None:
    with pytest.raises(OLMoConfigurationError, match="sequence_length"):
        build_pretrain_config(
            _opts(tmp_path, sequence_length=sequence_length),
            [],
            build_cubit_olmo3_1b,
            variant="cubit-fa3",
        )
