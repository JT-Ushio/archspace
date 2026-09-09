"""30B FP8 AdamW / Down-silu ablation."""
import argparse
from typing import List

from olmo_core.script_utils import ExperimentConfig, main
from olmo_mose import MoSENonlinearity
from _adamw_common import build_adamw_down_config


def build_config(opts: argparse.Namespace, overrides: List[str]) -> ExperimentConfig:
    return build_adamw_down_config(
        opts,
        overrides,
        variant="down-activation-silu-adamw-fp8-30b",
        down_nonlinearity=MoSENonlinearity.silu,
    )


if __name__ == "__main__":
    main(build_config)
