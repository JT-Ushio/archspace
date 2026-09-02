"""OLMo3-1B MoSE: Gate/Up SiLU and Down RMSNorm, all using the r2=512 branch."""

import argparse
import sys
from pathlib import Path
from typing import List

from olmo_core.optim import WSD
from olmo_core.script_utils import ExperimentConfig, main
from olmo_core.train import Duration

# Reuse the common builders without keeping this experiment in the shared
# cfgs directory.
CFGS_DIR = Path(__file__).resolve().parents[2] / "cfgs"
if str(CFGS_DIR) not in sys.path:
    sys.path.insert(0, str(CFGS_DIR))

from _models import build_mose_olmo3_1b  # noqa: E402
from _pretrain_common import build_pretrain_config  # noqa: E402
from olmo_mose import (  # noqa: E402
    MoSENonlinearity,
    SwiGLUChannelControl,
    SwiGLUChannelControlScope,
)

SEQUENCE_LENGTH = 8192
GLOBAL_BATCH_SIZE = 256 * SEQUENCE_LENGTH
RANK_MICROBATCH_SIZE = 8 * SEQUENCE_LENGTH
MAX_TOKENS = 30_000_000_000
CHECKPOINT_INTERVAL = round(7_500_000_000 / GLOBAL_BATCH_SIZE)
MUON_LR = 5e-3


def build_config(opts: argparse.Namespace, overrides: List[str]) -> ExperimentConfig:
    # This recipe is intentionally fixed at 8K context. The global batch size
    # is expressed in tokens by OLMo-core: 256 sequences * 8192 tokens.
    opts.sequence_length = SEQUENCE_LENGTH
    config = build_pretrain_config(
        opts,
        [],
        lambda tokenizer: build_mose_olmo3_1b(
            tokenizer,
            SwiGLUChannelControl.standard,
            control_scope=SwiGLUChannelControlScope.none,
            r1=0,
            r2=512,
            down_r1=0,
            down_r2=512,
            gate_nonlinearity=MoSENonlinearity.silu,
            up_nonlinearity=MoSENonlinearity.silu,
            down_nonlinearity=MoSENonlinearity.rms_norm,
            rms_norm_learnable_weight=False,
        ),
        variant="mose-r1-0-r2-512-down-r1-0-down-r2-512-gate-silu-up-silu-down-rmsnorm",
    )

    config.data_loader.global_batch_size = GLOBAL_BATCH_SIZE
    config.train_module.rank_microbatch_size = RANK_MICROBATCH_SIZE
    config.train_module.optim.lr = MUON_LR
    config.train_module.scheduler = WSD(warmup=2_000, decay_fraction=0.2)
    config.train_module.float8_config.enabled = True
    config.trainer.max_duration = Duration.tokens(MAX_TOKENS)

    checkpointer = config.trainer.callbacks["checkpointer"]
    checkpointer.save_interval = CHECKPOINT_INTERVAL
    checkpointer.max_checkpoints = None

    return config.merge(overrides)


if __name__ == "__main__":
    main(build_config)
