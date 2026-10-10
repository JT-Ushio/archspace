"""Dense-parameter-matched MoSE recipes for the OLMo3-1B comparison."""

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

# Both F0 and F1 have exactly this projection-parameter delta per Transformer
# block relative to native dense Q/K/V/O + SwiGLU. With 16 blocks this is
# 524,288 parameters, or 0.03531% below the 1,484,916,736-parameter baseline.
EXPECTED_PROJECTION_DELTA_PER_BLOCK = -32_768
MAX_RELATIVE_PROJECTION_MISMATCH = 1e-3


def _rank(value: int | None, fallback: int) -> int:
    return fallback if value is None else value


def _validate_dense_parameter_match(config: ExperimentConfig) -> None:
    """Fail before allocation if merged ranks no longer match the dense budget."""

    model = config.model
    d_model = model.d_model
    attention = model.block.sequence_mixer
    feed_forward = model.block.feed_forward

    n_heads = attention.n_heads
    n_kv_heads = attention.n_kv_heads or n_heads
    head_dim = attention.head_dim or d_model // n_heads
    q_size = n_heads * head_dim
    kv_size = n_kv_heads * head_dim

    q_rank = _rank(attention.q_rank, attention.rank)
    k_rank = _rank(attention.k_rank, attention.rank)
    v_rank = _rank(attention.v_rank, attention.rank)
    o_rank = _rank(attention.o_rank, attention.rank)
    scope = LowRankAttentionSharingScope(attention.share_scope)

    input_projection_params = {
        LowRankAttentionSharingScope.none: d_model
        * (q_rank + k_rank + v_rank + o_rank),
        LowRankAttentionSharingScope.qk: d_model
        * (q_rank + v_rank + o_rank),
        LowRankAttentionSharingScope.kv: d_model
        * (q_rank + k_rank + o_rank),
        LowRankAttentionSharingScope.qkv: d_model * (q_rank + o_rank),
    }[scope]
    attention_projection_params = input_projection_params + (
        q_rank * q_size
        + k_rank * kv_size
        + v_rank * kv_size
        + o_rank * d_model
    )
    dense_attention_projection_params = (
        d_model * q_size
        + 2 * d_model * kv_size
        + q_size * d_model
    )

    gate_up_u_count = 1 if feed_forward.share_gate_up_subspace else 2
    gate_up_rank = feed_forward.r1 + feed_forward.r2
    down_rank = feed_forward.down_r1 + feed_forward.down_r2
    feed_forward_projection_params = gate_up_rank * (
        gate_up_u_count * d_model + 2 * feed_forward.hidden_size
    ) + down_rank * (feed_forward.hidden_size + d_model)
    dense_feed_forward_projection_params = (
        3 * d_model * feed_forward.hidden_size
    )

    candidate = attention_projection_params + feed_forward_projection_params
    dense = dense_attention_projection_params + dense_feed_forward_projection_params
    delta = candidate - dense
    relative_mismatch = abs(delta) / dense
    if delta != EXPECTED_PROJECTION_DELTA_PER_BLOCK:
        raise ValueError(
            "dense-matched ranks changed: expected per-block projection delta "
            f"{EXPECTED_PROJECTION_DELTA_PER_BLOCK:,}, got {delta:,}"
        )
    if relative_mismatch > MAX_RELATIVE_PROJECTION_MISMATCH:
        raise ValueError(
            "projection parameter mismatch exceeds 0.1%: "
            f"{candidate:,} vs dense {dense:,}"
        )


def make_build_config(
    *,
    variant: str,
    share_gate_up_subspace: bool,
    gate_up_rank: int,
    attention_share_scope: LowRankAttentionSharingScope,
    attention_q_rank: int,
    attention_k_rank: int,
    attention_v_rank: int,
    attention_o_rank: int,
    down_rank: int = 1632,
) -> Callable[[argparse.Namespace, List[str]], ExperimentConfig]:
    """Create an F0/F1 config while keeping all non-architecture settings fixed."""

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
                gate_nonlinearity=MoSENonlinearity.silu,
                up_nonlinearity=MoSENonlinearity.silu,
                down_nonlinearity=MoSENonlinearity.rms_norm,
                rms_norm_learnable_weight=False,
                share_gate_up_subspace=share_gate_up_subspace,
                attention_low_rank_enabled=True,
                attention_rank=attention_q_rank,
                attention_q_rank=attention_q_rank,
                attention_k_rank=attention_k_rank,
                attention_v_rank=attention_v_rank,
                attention_o_rank=attention_o_rank,
                attention_q_nonlinearity=MoSENonlinearity.identity,
                attention_k_nonlinearity=MoSENonlinearity.identity,
                attention_v_nonlinearity=MoSENonlinearity.identity,
                attention_o_nonlinearity=MoSENonlinearity.identity,
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
        _validate_dense_parameter_match(config)
        return config

    return build_config
