"""OLMo3-1B precision ablation: Down SiLU with FP8 enabled."""

import argparse
from typing import List

from olmo_core.script_utils import ExperimentConfig, main

from _common import build_precision_config
from olmo_mose import MoSENonlinearity


def build_config(opts: argparse.Namespace, overrides: List[str]) -> ExperimentConfig:
    return build_precision_config(
        opts,
        overrides,
        variant="precision-down-silu-fp8",
        down_nonlinearity=MoSENonlinearity.silu,
        float8_enabled=True,
    )


if __name__ == "__main__":
    main(build_config)
