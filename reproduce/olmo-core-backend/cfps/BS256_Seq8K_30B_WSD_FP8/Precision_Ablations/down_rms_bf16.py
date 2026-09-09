"""OLMo3-1B precision ablation: Down RMSNorm with FP8 disabled."""

import argparse
from typing import List

from olmo_core.script_utils import ExperimentConfig, main

from _common import build_precision_config
from olmo_mose import MoSENonlinearity


def build_config(opts: argparse.Namespace, overrides: List[str]) -> ExperimentConfig:
    return build_precision_config(
        opts,
        overrides,
        variant="precision-down-rms-bf16",
        down_nonlinearity=MoSENonlinearity.rms_norm,
        float8_enabled=False,
    )


if __name__ == "__main__":
    main(build_config)
