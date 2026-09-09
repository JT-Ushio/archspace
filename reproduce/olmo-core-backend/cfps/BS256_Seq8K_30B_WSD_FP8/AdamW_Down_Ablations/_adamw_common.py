"""AdamW controls matched to the 30B FP8 Down activation experiments."""
import argparse
import sys
from pathlib import Path
from typing import List

from olmo_core.optim import AdamWConfig, OptimGroupOverride
from olmo_core.script_utils import ExperimentConfig
from olmo_mose import MoSENonlinearity

BACKEND_DIR = Path(__file__).resolve().parents[3]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from cfps.BS256_Seq8K_30B_WSD_FP8.Down_Identity_Ablations._common import (  # noqa: E402
    build_down_activation_config,
)


def build_adamw_down_config(
    opts: argparse.Namespace,
    overrides: List[str],
    *,
    variant: str,
    down_nonlinearity: MoSENonlinearity,
) -> ExperimentConfig:
    if down_nonlinearity not in (MoSENonlinearity.silu, MoSENonlinearity.rms_norm):
        raise ValueError("This paired experiment supports only Down-SiLU / Down-RMSNorm")
    config = build_down_activation_config(
        opts, [], variant=variant, down_nonlinearity=down_nonlinearity,
    )
    # Replace the optimizer, not just the Muon learning rate. All trainable
    # parameters now use AdamW, including embeddings, norms and the LM head.
    # Match the baseline's no-decay categories for this bias-free architecture.
    config.train_module.optim = AdamWConfig(
        lr=1e-3,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.033,
        group_overrides=[
            OptimGroupOverride(
                params=["embeddings.*", "*norm*.weight"],
                opts={"weight_decay": 0.0},
            ),
        ],
    )
    # Apply overrides only AFTER replacing Muon, so the CLI edits AdamW.
    return config.merge(overrides)
