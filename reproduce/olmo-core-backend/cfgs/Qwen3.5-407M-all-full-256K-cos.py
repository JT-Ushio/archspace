"""Full Attention / 256K control with cosine decay to the same final learning rate."""

import argparse
from typing import List

from olmo_core.optim import CosWithWarmup, SchedulerUnits
from olmo_core.script_utils import ExperimentConfig, main

from _qwen35_407m_entry import build_entry_config
from _qwen35_407m_models import build_qwen35_407m_all_full
from _qwen35_407m_pretrain import LR_FINAL_FACTOR, WARMUP_TOKENS


def build_config(opts: argparse.Namespace, overrides: List[str]) -> ExperimentConfig:
    """Reuse the all-full 256K recipe with a cosine scheduler and a separate W&B group."""
    config = build_entry_config(
        opts,
        [],
        model_builder=build_qwen35_407m_all_full,
        variant="all-full-cos",
        profile_name="256K",
    )
    config.train_module.scheduler = CosWithWarmup(
        units=SchedulerUnits.tokens,
        warmup=WARMUP_TOKENS,
        alpha_f=LR_FINAL_FACTOR,
    )
    return config.merge(overrides)


if __name__ == "__main__":
    main(build_config)
