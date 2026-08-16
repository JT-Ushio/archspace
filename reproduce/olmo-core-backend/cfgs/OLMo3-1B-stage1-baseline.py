"""OLMo3-1B stage-1 pretraining control using FlashAttention 3."""

import argparse
from typing import List

from olmo_core.script_utils import ExperimentConfig, main

from _models import build_olmo3_1b
from _pretrain_common import build_pretrain_config


def build_config(opts: argparse.Namespace, overrides: List[str]) -> ExperimentConfig:
    return build_pretrain_config(
        opts,
        overrides,
        build_olmo3_1b,
        variant="baseline-fa3",
    )


if __name__ == "__main__":
    main(build_config)
