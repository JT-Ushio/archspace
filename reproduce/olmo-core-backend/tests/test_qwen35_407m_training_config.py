import sys
from argparse import Namespace
from pathlib import Path

import pytest

from olmo_core.config import DType
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.optim import SchedulerUnits, WSDSqrtDecay
from olmo_core.optim.muon import MuonAdjustLRStrategy
from olmo_core.script_utils import ExperimentConfig
from olmo_core.train import DurationUnit
from olmo_ahn import SerializableMuonConfig

CFGS_DIR = Path(__file__).parents[1] / "cfgs"
sys.path.insert(0, str(CFGS_DIR))

from _qwen35_407m_models import build_qwen35_407m_gdn3_full1  # noqa: E402
from _qwen35_407m_pretrain import (  # noqa: E402
    LR_FINAL_FACTOR,
    LR_MAX,
    RANK_MICROBATCH_SIZE,
    SEQUENCE_LENGTH,
    TRAINING_PROFILES,
    WARMUP_TOKENS,
    build_qwen35_407m_pretrain_config,
)


def _opts(tmp_path: Path, *, sequence_length: int | None = None) -> Namespace:
    return Namespace(
        sequence_length=sequence_length,
        data_root=str(tmp_path / "data"),
        work_dir=str(tmp_path / "work"),
        save_folder=str(tmp_path / "checkpoints"),
        name="qwen35-407m-config-test",
    )


@pytest.mark.parametrize("profile_name", TRAINING_PROFILES)
def test_qwen35_407m_training_profiles_round_trip(
    tmp_path: Path,
    profile_name: str,
) -> None:
    profile = TRAINING_PROFILES[profile_name]
    config = build_qwen35_407m_pretrain_config(
        _opts(tmp_path),
        [],
        build_qwen35_407m_gdn3_full1,
        variant="gdn3-full1",
        profile_name=profile_name,
    )
    restored = ExperimentConfig.from_dict(config.as_config_dict())

    assert restored.dataset.sequence_length == SEQUENCE_LENGTH
    assert restored.dataset.tokenizer.padded_vocab_size() == 100_352
    assert restored.data_loader.global_batch_size == profile.global_batch_size
    assert restored.train_module.rank_microbatch_size == RANK_MICROBATCH_SIZE
    assert restored.train_module.max_sequence_length == SEQUENCE_LENGTH
    assert isinstance(restored.train_module.optim, SerializableMuonConfig)
    assert restored.train_module.optim.flatten is False
    assert restored.train_module.optim.lr == LR_MAX
    assert restored.train_module.optim.weight_decay == 0.1
    assert restored.train_module.optim.mu == 0.95
    assert restored.train_module.optim.adjust_lr == MuonAdjustLRStrategy.rms_norm
    assert restored.train_module.optim.use_polar_express is False
    assert restored.train_module.dp_config.param_dtype == DType.bfloat16
    assert restored.trainer.max_duration.unit == DurationUnit.steps
    assert restored.trainer.max_duration.value == profile.max_steps

    scheduler = restored.train_module.scheduler
    assert isinstance(scheduler, WSDSqrtDecay)
    assert scheduler.units == SchedulerUnits.tokens
    assert scheduler.warmup == WARMUP_TOKENS
    assert scheduler.decay_fraction == 0.1
    assert scheduler.decay_min_lr_ratio == LR_FINAL_FACTOR

    max_tokens = profile.max_tokens
    decay_start = max_tokens - round(max_tokens * scheduler.decay_fraction)
    decay_middle = decay_start + round((max_tokens - decay_start) / 4)
    assert scheduler.get_lr(LR_MAX, 0, max_tokens) == 0.0
    assert scheduler.get_lr(LR_MAX, WARMUP_TOKENS, max_tokens) == LR_MAX
    assert scheduler.get_lr(LR_MAX, decay_start - 1, max_tokens) == LR_MAX
    assert scheduler.get_lr(LR_MAX, decay_middle, max_tokens) == pytest.approx(
        LR_MAX * (1 + LR_FINAL_FACTOR) / 2,
        rel=1e-8,
    )
    assert scheduler.get_lr(LR_MAX, max_tokens, max_tokens) == LR_MAX * LR_FINAL_FACTOR


def test_qwen35_407m_rejects_non_8k_sequence_length(tmp_path: Path) -> None:
    with pytest.raises(OLMoConfigurationError, match="sequence_length=8192"):
        build_qwen35_407m_pretrain_config(
            _opts(tmp_path, sequence_length=4_096),
            [],
            build_qwen35_407m_gdn3_full1,
            variant="gdn3-full1",
            profile_name="256K",
        )
