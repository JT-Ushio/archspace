"""Rank-512-budget-matched combinations of the strongest MoSE ablations."""

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
RANK_MICROBATCH_SIZE = 4 * SEQUENCE_LENGTH
MAX_TOKENS = 30_000_000_000
CHECKPOINT_INTERVAL = round((MAX_TOKENS * 0.125) / GLOBAL_BATCH_SIZE)
MUON_LR = 5e-3

# Projection budget of the fully unshared rank-512 model. Every combination
# below matches this value exactly, including small compensating rank changes.
REFERENCE_RANK = 512
REFERENCE_PROJECTION_PARAMS_PER_BLOCK = 24_117_248


def _validate_rank512_budget(config: ExperimentConfig) -> None:
    model = config.model
    d_model = model.d_model
    attention = model.block.sequence_mixer
    feed_forward = model.block.feed_forward

    q_rank = attention.q_rank or attention.rank
    k_rank = attention.k_rank or attention.rank
    v_rank = attention.v_rank or attention.rank
    o_rank = attention.o_rank or attention.rank
    scope = LowRankAttentionSharingScope(attention.share_scope)

    input_projection_params = {
        LowRankAttentionSharingScope.none: d_model
        * (q_rank + k_rank + v_rank + o_rank),
        LowRankAttentionSharingScope.kv: d_model * (q_rank + k_rank + o_rank),
    }.get(scope)
    if input_projection_params is None:
        raise ValueError(f"unsupported sharing scope for this sweep: {scope}")

    # OLMo3-1B has q_size == kv_size == d_model.
    attention_params = input_projection_params + d_model * (
        q_rank + k_rank + v_rank + o_rank
    )
    gate_up_rank = feed_forward.r1 + feed_forward.r2
    down_rank = feed_forward.down_r1 + feed_forward.down_r2
    gate_up_u_count = 1 if feed_forward.share_gate_up_subspace else 2
    feed_forward_params = gate_up_rank * (
        gate_up_u_count * d_model + 2 * feed_forward.hidden_size
    ) + down_rank * (feed_forward.hidden_size + d_model)

    actual = attention_params + feed_forward_params
    if actual != REFERENCE_PROJECTION_PARAMS_PER_BLOCK:
        raise ValueError(
            "rank-512 budget mismatch: expected "
            f"{REFERENCE_PROJECTION_PARAMS_PER_BLOCK:,} projection parameters "
            f"per block, got {actual:,}"
        )


def make_build_config(
    *,
    variant: str,
    share_gate_up_subspace: bool,
    gate_up_rank: int,
    attention_share_scope: LowRankAttentionSharingScope,
    q_rank: int,
    k_rank: int,
    v_rank: int,
    o_rank: int,
    down_rank: int,
    q_nonlinearity: MoSENonlinearity = MoSENonlinearity.identity,
    k_nonlinearity: MoSENonlinearity = MoSENonlinearity.silu,
    v_nonlinearity: MoSENonlinearity = MoSENonlinearity.identity,
    o_nonlinearity: MoSENonlinearity = MoSENonlinearity.identity,
    gate_nonlinearity: MoSENonlinearity = MoSENonlinearity.identity,
    up_nonlinearity: MoSENonlinearity = MoSENonlinearity.identity,
) -> Callable[[argparse.Namespace, List[str]], ExperimentConfig]:
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
                r2=gate_up_rank,
                down_r1=0,
                down_r2=down_rank,
                gate_nonlinearity=gate_nonlinearity,
                up_nonlinearity=up_nonlinearity,
                down_nonlinearity=MoSENonlinearity.rms_norm,
                rms_norm_learnable_weight=False,
                share_gate_up_subspace=share_gate_up_subspace,
                attention_low_rank_enabled=True,
                attention_rank=REFERENCE_RANK,
                attention_q_rank=q_rank,
                attention_k_rank=k_rank,
                attention_v_rank=v_rank,
                attention_o_rank=o_rank,
                attention_q_nonlinearity=q_nonlinearity,
                attention_k_nonlinearity=k_nonlinearity,
                attention_v_nonlinearity=v_nonlinearity,
                attention_o_nonlinearity=o_nonlinearity,
                attention_rms_norm_learnable_weight=False,
                attention_share_scope=attention_share_scope,
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

        config = config.merge(overrides)
        _validate_rank512_budget(config)
        return config

    return build_config
