import sys
from argparse import Namespace
from pathlib import Path

import pytest

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.script_utils import ExperimentConfig
from olmo_mose import (
    ChannelControlledFeedForwardConfig,
    MoSENonlinearity,
    MoSESwiGLUConfig,
    SwiGLUChannelControl,
)


CFGS_DIR = Path(__file__).parents[1] / "cfgs"
sys.path.insert(0, str(CFGS_DIR))

from _models import build_mose_olmo3_1b, build_olmo3_1b  # noqa: E402
from _pretrain_common import (  # noqa: E402
    N_BATCH_PER_GPU,
    build_pretrain_config,
    get_mose_cli_parser,
)


def test_mose_cli_parser_accepts_layer_range() -> None:
    opts = get_mose_cli_parser().parse_args(
        [
            "--save-folder=/tmp/checkpoints",
            "--mose-start-layer=2",
            "--mose-end-layer=15",
        ]
    )

    assert opts.mose_start_layer == 2
    assert opts.mose_end_layer == 15


def test_experiment_config_round_trips(tmp_path: Path) -> None:
    opts = Namespace(
        sequence_length=None,
        data_root=str(tmp_path / "data"),
        work_dir=str(tmp_path / "work"),
        save_folder=str(tmp_path / "checkpoints"),
        name="config-round-trip",
    )
    config = build_pretrain_config(
        opts,
        [],
        lambda tokenizer: build_olmo3_1b(tokenizer, SwiGLUChannelControl.situ),
        variant="situ",
    )

    restored = ExperimentConfig.from_dict(config.as_config_dict())
    checkpointer = restored.trainer.callbacks["checkpointer"]

    assert checkpointer.save_interval == 2500
    assert checkpointer.ephemeral_save_interval is None
    assert isinstance(restored.model.block.feed_forward, ChannelControlledFeedForwardConfig)


def test_mose_experiment_config_round_trips_rank_overrides(tmp_path: Path) -> None:
    opts = Namespace(
        sequence_length=None,
        data_root=str(tmp_path / "data"),
        work_dir=str(tmp_path / "work"),
        save_folder=str(tmp_path / "checkpoints"),
        name="mose-config-round-trip",
    )
    config = build_pretrain_config(
        opts,
        [
            "model.block.feed_forward.r1=64",
            "model.block.feed_forward.r2=32",
            "model.block.feed_forward.down_r1=16",
            "model.block.feed_forward.down_r2=0",
            "model.block.feed_forward.gate_nonlinearity=rms_norm",
            "model.block.feed_forward.up_nonlinearity=rms_norm",
            "model.block.feed_forward.down_nonlinearity=silu",
            "model.block.feed_forward.rms_norm_learnable_weight=true",
        ],
        lambda tokenizer: build_mose_olmo3_1b(
            tokenizer,
            SwiGLUChannelControl.asymmetric_rational_clip,
        ),
        variant="mose-asymmetric-rational-clip",
    )

    restored = ExperimentConfig.from_dict(config.as_config_dict())
    feed_forward = restored.model.block.feed_forward

    assert isinstance(feed_forward, MoSESwiGLUConfig)
    assert (feed_forward.r1, feed_forward.r2) == (64, 32)
    assert (feed_forward.down_r1, feed_forward.down_r2) == (16, 0)
    assert feed_forward.control == SwiGLUChannelControl.asymmetric_rational_clip
    assert feed_forward.gate_nonlinearity == MoSENonlinearity.rms_norm
    assert feed_forward.up_nonlinearity == MoSENonlinearity.rms_norm
    assert feed_forward.down_nonlinearity == MoSENonlinearity.silu
    assert feed_forward.rms_norm_learnable_weight is True


def test_sequence_length_updates_rank_microbatch_size(tmp_path: Path) -> None:
    opts = Namespace(
        sequence_length=8192,
        data_root=str(tmp_path / "data"),
        work_dir=str(tmp_path / "work"),
        save_folder=str(tmp_path / "checkpoints"),
        name="longer-sequence",
    )

    config = build_pretrain_config(
        opts,
        [],
        lambda tokenizer: build_olmo3_1b(tokenizer, SwiGLUChannelControl.standard),
        variant="baseline",
    )

    assert config.train_module.rank_microbatch_size == N_BATCH_PER_GPU * 8192


@pytest.mark.parametrize("sequence_length", [0, 3072, 12_288])
def test_invalid_sequence_length_fails_during_config_build(
    tmp_path: Path,
    sequence_length: int,
) -> None:
    opts = Namespace(
        sequence_length=sequence_length,
        data_root=str(tmp_path / "data"),
        work_dir=str(tmp_path / "work"),
        save_folder=str(tmp_path / "checkpoints"),
        name="invalid-sequence",
    )

    with pytest.raises(OLMoConfigurationError, match="sequence_length"):
        build_pretrain_config(
            opts,
            [],
            lambda tokenizer: build_olmo3_1b(tokenizer, SwiGLUChannelControl.standard),
            variant="baseline",
        )
