"""All-linear control with ONLY Q low-rank intermediate changed to SiLU."""
import argparse
import sys
from pathlib import Path
from typing import List

from olmo_core.script_utils import ExperimentConfig, main
from olmo_mose import MoSENonlinearity

BACKEND_DIR = Path(__file__).resolve().parents[3]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from cfps.BS256_Seq8K_30B_WSD_FP8.All_Linear_Ablation.all_linear_fp8 import (  # noqa: E402
    build_config as build_all_linear_config,
)


def build_config(opts: argparse.Namespace, overrides: List[str]) -> ExperimentConfig:
    config = build_all_linear_config(opts, [])
    config.model.block.sequence_mixer.q_nonlinearity = MoSENonlinearity.silu
    config.trainer.callbacks["wandb"].group = "all-linear-attention-single-silu-fp8-30b"
    return config.merge(overrides)


if __name__ == "__main__":
    main(build_config)

