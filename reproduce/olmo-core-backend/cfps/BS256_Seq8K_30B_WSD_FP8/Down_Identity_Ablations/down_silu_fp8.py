"""30B FP8 Down activation ablation: silu."""
import argparse
from typing import List

from olmo_core.script_utils import ExperimentConfig, main
from olmo_mose import MoSENonlinearity
from _common import build_down_activation_config


def build_config(opts: argparse.Namespace, overrides: List[str]) -> ExperimentConfig:
    return build_down_activation_config(
        opts,
        overrides,
        variant="down-activation-silu-fp8-30b",
        down_nonlinearity=MoSENonlinearity.silu,
    )


if __name__ == "__main__":
    main(build_config)
