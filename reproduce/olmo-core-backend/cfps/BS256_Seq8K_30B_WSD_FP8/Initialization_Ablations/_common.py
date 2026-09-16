"""Same Down-RMS control; only initialization strategy / run namespace differ."""
from dataclasses import fields
from pathlib import Path
import sys

from olmo_core.nn.transformer import TransformerConfig
from olmo_mose import MoSENonlinearity
from olmo_mose.svd_initialization import SVDInitTransformerConfig

BACKEND = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BACKEND))
from cfps.BS256_Seq8K_30B_WSD_FP8.Down_Identity_Ablations._common import build_down_activation_config


def build_init_config(opts, overrides, mode):
    config = build_down_activation_config(
        opts, [], variant=f"init-{mode.replace('_', '-')}-down-rms-fp8-30b",
        down_nonlinearity=MoSENonlinearity.rms_norm)
    base = config.model
    config.model = SVDInitTransformerConfig(
        **{f.name: getattr(base, f.name) for f in fields(TransformerConfig)},
        factor_init=mode)
    config.trainer.callbacks["wandb"].group = "initialization-down-rms-fp8-30b"
    return config.merge(overrides)
