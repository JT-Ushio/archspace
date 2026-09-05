"""Shared recipe for parameter-matched attention-sharing ablations."""

import argparse
import sys
from pathlib import Path
from typing import Callable, List

from olmo_core.optim import WSD
from olmo_core.script_utils import ExperimentConfig
from olmo_core.train import Duration

CFGS_DIR = Path(__file__).resolve().parents[3] / "cfgs"
if str(CFGS_DIR) not in sys.path:
    sys.path.insert(0, str(CFGS_DIR))

from _models import build_mose_olmo3_1b  # noqa: E402
from _pretrain_common import build_pretrain_config  # noqa: E402
from olmo_mose import (  # noqa: E402
    LowRankAttentionSharingScope,
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
DOWN_RANK = 512
ATTENTION_FALLBACK_RANK = 512


def make_build_config(
    *,
    variant: str,
    share_gate_up_subspace: bool,
    up_gate_rank: int,
    attention_share_scope: LowRankAttentionSharingScope,
    attention_q_rank: int,
    attention_k_rank: int,
    attention_v_rank: int,
    attention_o_rank: int,
) -> Callable[[argparse.Namespace, List[str]], ExperimentConfig]:
    """Create a config builder while keeping the recipe identical across runs."""

    def build_config(opts: argparse.Namespace, overrides: List[str]) -> ExperimentConfig:
        opts.sequence_length = SEQUENCE_LENGTH
        config = build_pretrain_config(
            opts,
            [],
            lambda tokenizer: build_mose_olmo3_1b(
                tokenizer,
                SwiGLUChannelControl.standard,
                control_scope=SwiGLUChannelControlScope.none,
                r1=0,
                r2=up_gate_rank,
                down_r1=0,
                down_r2=DOWN_RANK,
                gate_nonlinearity=MoSENonlinearity.silu,
                up_nonlinearity=MoSENonlinearity.silu,
                down_nonlinearity=MoSENonlinearity.rms_norm,
                rms_norm_learnable_weight=False,
                share_gate_up_subspace=share_gate_up_subspace,
                attention_low_rank_enabled=True,
                attention_rank=ATTENTION_FALLBACK_RANK,
                attention_q_rank=attention_q_rank,
                attention_k_rank=attention_k_rank,
                attention_v_rank=attention_v_rank,
                attention_o_rank=attention_o_rank,
                attention_share_scope=attention_share_scope,
                attention_q_nonlinearity=MoSENonlinearity.silu,
                attention_k_nonlinearity=MoSENonlinearity.silu,
                attention_v_nonlinearity=MoSENonlinearity.silu,
                attention_o_nonlinearity=MoSENonlinearity.silu,
                attention_rms_norm_learnable_weight=False,
            ),
            variant=variant,
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

    return build_config
