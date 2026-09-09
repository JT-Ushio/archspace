"""Identity between all low-rank factors; retain the primary SwiGLU gate.

Q/K/V/O, Gate/Up and Down keep their original matrices and initialization.
This does not remove attention softmax, outer RMSNorm or SiLU(gate) * up.
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
        opts, [], variant="all-lowrank-linear-fp8-30b",
        down_nonlinearity=MoSENonlinearity.identity,
    )
    attention = config.model.block.sequence_mixer
    for projection in ("q", "k", "v", "o"):
        setattr(attention, f"{projection}_nonlinearity", MoSENonlinearity.identity)
    config.model.block.feed_forward.gate_nonlinearity = MoSENonlinearity.identity
    config.model.block.feed_forward.up_nonlinearity = MoSENonlinearity.identity
    return config.merge(overrides)


if __name__ == "__main__":
    main(build_config)
