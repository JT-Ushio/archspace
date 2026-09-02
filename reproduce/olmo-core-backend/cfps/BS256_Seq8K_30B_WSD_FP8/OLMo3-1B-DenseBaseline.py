"""OLMo3-1B dense baseline with the shared 30B FP8 training recipe."""

import argparse
import sys
from pathlib import Path
from typing import List

from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import WSD
from olmo_core.script_utils import ExperimentConfig, main
from olmo_core.train import Duration

# The shared stage-1 recipe lives in the sibling cfgs directory.
CFGS_DIR = Path(__file__).resolve().parents[2] / "cfgs"
if str(CFGS_DIR) not in sys.path:
    sys.path.insert(0, str(CFGS_DIR))

from _pretrain_common import build_pretrain_config

SEQUENCE_LENGTH = 8192
GLOBAL_BATCH_SIZE = 256 * SEQUENCE_LENGTH
RANK_MICROBATCH_SIZE = 8 * SEQUENCE_LENGTH
MAX_TOKENS = 30_000_000_000
CHECKPOINT_INTERVAL = round(7_500_000_000 / GLOBAL_BATCH_SIZE)
MUON_LR = 5e-3


def build_config(opts: argparse.Namespace, overrides: List[str]) -> ExperimentConfig:
    opts.sequence_length = SEQUENCE_LENGTH
    config = build_pretrain_config(
        opts,
        [],
        lambda tokenizer: TransformerConfig.olmo3_1B(
            vocab_size=tokenizer.padded_vocab_size(),
            attn_backend=AttentionBackendName.flash_3,
        ),
        variant="dense-baseline",
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
