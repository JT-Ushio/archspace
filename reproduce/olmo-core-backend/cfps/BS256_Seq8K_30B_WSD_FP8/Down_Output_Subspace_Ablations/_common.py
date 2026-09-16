"""Down-Identity control with static/dynamic output rotation, otherwise unchanged."""
from pathlib import Path
import sys

from olmo_mose import MoSENonlinearity
from olmo_mose.output_subspace import DownOutputRotation

BACKEND = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BACKEND))
from cfps.BS256_Seq8K_30B_WSD_FP8.Down_Identity_Ablations._common import build_down_activation_config


def build_rotation_config(opts, overrides, mode):
    config = build_down_activation_config(
        opts, [], variant=f"down-output-subspace-{mode}-fp8-30b",
        down_nonlinearity=MoSENonlinearity.identity)
    ff = config.model.block.feed_forward
    ff.down_output_rotation = DownOutputRotation(mode)
    ff.down_rotation_groups = 32
    ff.down_rotation_a_max = 0.5
    config.trainer.callbacks["wandb"].group = "down-output-subspace-fp8-30b"
    config = config.merge(overrides)
    # Validate merged settings before scheduling/model allocation, including widths.
    config.model.block.feed_forward.num_params(config.model.d_model)
    return config
