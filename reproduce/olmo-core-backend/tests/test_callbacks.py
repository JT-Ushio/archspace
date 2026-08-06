from unittest.mock import Mock, patch

import pytest
from olmo_core.train import TrainerConfig

from olmo_mla_morphnorm import MaxLogitsWandBCallback


def test_max_logits_wandb_callback_filters_only_max_logits_between_intervals() -> None:
    callback = MaxLogitsWandBCallback(max_logits_log_interval=10)
    callback._wandb = Mock()
    metrics = {
        "train/loss": 1.0,
        "train/attention max logit": 2.0,
        "train/block 00/attention max logit/head 00": 3.0,
    }

    with patch("olmo_core.train.callbacks.wandb.get_rank", return_value=0):
        callback.log_metrics(9, metrics)

    callback.wandb.log.assert_called_once_with({"train/loss": 1.0}, step=9)
    assert len(metrics) == 3


def test_max_logits_wandb_callback_logs_all_metrics_on_interval() -> None:
    callback = MaxLogitsWandBCallback(max_logits_log_interval=10)
    callback._wandb = Mock()
    metrics = {
        "train/loss": 1.0,
        "train/attention max logit": 2.0,
    }

    with patch("olmo_core.train.callbacks.wandb.get_rank", return_value=0):
        callback.log_metrics(10, metrics)

    callback.wandb.log.assert_called_once_with(metrics, step=10)


def test_max_logits_wandb_callback_rejects_invalid_interval() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        MaxLogitsWandBCallback(max_logits_log_interval=0)


def test_max_logits_wandb_interval_can_be_overridden_from_config() -> None:
    config = TrainerConfig(save_folder="/tmp/checkpoints").with_callback(
        "wandb", MaxLogitsWandBCallback(max_logits_log_interval=100)
    )

    merged = config.merge(["callbacks.wandb.max_logits_log_interval=25"])

    callback = merged.callbacks["wandb"]
    assert isinstance(callback, MaxLogitsWandBCallback)
    assert callback.max_logits_log_interval == 25
