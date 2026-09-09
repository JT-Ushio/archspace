"""30B FP8 AdamW / Down-rms ablation."""
import argparse
from typing import List

from olmo_core.script_utils import ExperimentConfig, main
from olmo_mose import MoSENonlinearity
from _adamw_common import build_adamw_down_config


def build_config(opts: argparse.Namespace, overrides: List[str]) -> ExperimentConfig:
    return build_adamw_down_config(
        opts,
        overrides,
        variant="down-activation-rms-adamw-fp8-30b",
        down_nonlinearity=MoSENonlinearity.rms_norm,
    )


if __name__ == "__main__":
    main(build_config)
