"""Small adapter used by the ten explicit Qwen3.5-407M launchers."""

import argparse
from typing import List

from olmo_core.script_utils import ExperimentConfig

from _qwen35_407m_pretrain import ModelBuilder, build_qwen35_407m_pretrain_config


def build_entry_config(
    opts: argparse.Namespace,
    overrides: List[str],
    *,
    model_builder: ModelBuilder,
    variant: str,
    profile_name: str,
) -> ExperimentConfig:
    return build_qwen35_407m_pretrain_config(
        opts,
        overrides,
        model_builder,
        variant=variant,
        profile_name=profile_name,
    )
