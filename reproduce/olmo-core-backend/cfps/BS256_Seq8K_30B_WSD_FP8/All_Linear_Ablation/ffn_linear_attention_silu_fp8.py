"""FFN low-rank factors use Identity; attention Q/K/V/O use SiLU.

Retain the primary SwiGLU SiLU(gate) * up, all outer norms and attention softmax.
Matched to all_linear_fp8 in ranks, parameter shapes and training settings.
"""
import argparse
import sys
from pathlib import Path
from typing import List

from olmo_core.script_utils import ExperimentConfig, main
from olmo_mose import MoSENonlinearity

BACKEND_DIR = Path(__file__).resolve().parents[3]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from cfps.BS256_Seq8K_30B_WSD_FP8.Down_Identity_Ablations._common import (  # noqa: E402
    build_down_activation_config,
)


def build_config(opts: argparse.Namespace, overrides: List[str]) -> ExperimentConfig:
    config = build_down_activation_config(
        opts, [], variant="ffn-linear-attention-silu-fp8-30b",
        down_nonlinearity=MoSENonlinearity.identity,
    )
    for projection in ("q", "k", "v", "o"):
        setattr(config.model.block.sequence_mixer, f"{projection}_nonlinearity", MoSENonlinearity.silu)
    for projection in ("gate", "up", "down"):
        setattr(config.model.block.feed_forward, f"{projection}_nonlinearity", MoSENonlinearity.identity)
    return config.merge(overrides)


if __name__ == "__main__":
    main(build_config)
