import runpy
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
    SwiGLUChannelControlScope,
)


CFGS_DIR = Path(__file__).parents[1] / "cfgs"
sys.path.insert(0, str(CFGS_DIR))

from _models import build_mose_olmo3_1b, build_olmo3_1b  # noqa: E402
from _pretrain_common import N_BATCH_PER_GPU, build_pretrain_config  # noqa: E402


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


@pytest.mark.parametrize(
    ("config_name", "expected_control", "expected_scope"),
    [
        (
            "OLMo3-1B-stage1-dpskv4-clip-up.py",
            SwiGLUChannelControl.dpskv4_clip,
            SwiGLUChannelControlScope.up,
        ),
        (
            "OLMo3-1B-stage1-dpskv4-clip-both.py",
            SwiGLUChannelControl.dpskv4_clip,
            SwiGLUChannelControlScope.both,
        ),
        (
            "OLMo3-1B-stage1-dpskv4-clip-gate-situ-up.py",
            SwiGLUChannelControl.dpskv4_clip_situ,
            SwiGLUChannelControlScope.both,
        ),
    ],
)
def test_dpskv4_experiment_entry_points(
    tmp_path: Path,
    config_name: str,
    expected_control: SwiGLUChannelControl,
    expected_scope: SwiGLUChannelControlScope,
) -> None:
    opts = Namespace(
        sequence_length=None,
        data_root=str(tmp_path / "data"),
        work_dir=str(tmp_path / "work"),
        save_folder=str(tmp_path / "checkpoints"),
        name="dpskv4-entry-point",
    )
    entry_point = runpy.run_path(str(CFGS_DIR / config_name))
    config = entry_point["build_config"](opts, [])
    feed_forward = config.model.block.feed_forward

    assert isinstance(feed_forward, ChannelControlledFeedForwardConfig)
    assert feed_forward.control == expected_control
    assert feed_forward.control_scope == expected_scope


def test_situ_rms_up_experiment_entry_point(tmp_path: Path) -> None:
    opts = Namespace(
        sequence_length=None,
        data_root=str(tmp_path / "data"),
        work_dir=str(tmp_path / "work"),
        save_folder=str(tmp_path / "checkpoints"),
        name="situ-rms-up-entry-point",
    )
    entry_point = runpy.run_path(str(CFGS_DIR / "OLMo3-1B-stage1-situ-rms-up.py"))
    config = entry_point["build_config"](
        opts,
        ["model.block.feed_forward.rms_beta_scale=3.5"],
    )
    feed_forward = config.model.block.feed_forward

    assert isinstance(feed_forward, ChannelControlledFeedForwardConfig)
    assert feed_forward.control == SwiGLUChannelControl.situ_rms
    assert feed_forward.control_scope == SwiGLUChannelControlScope.up
    assert feed_forward.rms_beta_scale == 3.5


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
            "model.block.feed_forward.control_scope=up",
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
    assert feed_forward.control_scope == SwiGLUChannelControlScope.up
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
