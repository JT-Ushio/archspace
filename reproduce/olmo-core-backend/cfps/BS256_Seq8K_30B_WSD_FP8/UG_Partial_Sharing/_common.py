"""Vary UG shared/private ranks while holding the other projections fixed."""

import sys
from dataclasses import replace
from pathlib import Path

from olmo_core.optim import WSD
from olmo_core.train import Duration

CFGS_DIR = Path(__file__).resolve().parents[3] / "cfgs"
if str(CFGS_DIR) not in sys.path:
    sys.path.insert(0, str(CFGS_DIR))

from _models import build_mose_olmo3_1b  # noqa: E402
from _pretrain_common import build_pretrain_config  # noqa: E402
from olmo_mose import (  # noqa: E402
    MoSENonlinearity,
    SwiGLUChannelControl,
    SwiGLUChannelControlScope,
)

# P1 rounds the fully shared endpoint down. Interior points and P0 match
# the independently rank-512 reference exactly, without changing O or Down.
VARIANTS = {
    "p0": (0, 512),
    "p1": (560, 0),
    "p2": (480, 80),
    "p3": (320, 224),
    "p4": (160, 368),
}
REFERENCE_PROJECTION_PARAMS_PER_BLOCK = 24_117_248


def variant_name(target):
    shared_rank, private_rank = VARIANTS[target]
    return f"ug-partial-{target}-s{shared_rank}-p{private_rank}-linear-down-rms"


def build_model(tokenizer, target):
    shared_rank, private_rank = VARIANTS[target]
    config = build_mose_olmo3_1b(
        tokenizer,
        SwiGLUChannelControl.standard,
        control_scope=SwiGLUChannelControlScope.none,
        r1=shared_rank,
        r2=private_rank,
        down_r1=0,
        down_r2=512,
        gate_nonlinearity=MoSENonlinearity.identity,
        up_nonlinearity=MoSENonlinearity.identity,
        down_nonlinearity=MoSENonlinearity.rms_norm,
        rms_norm_learnable_weight=False,
        share_gate_up_subspace=False,
        attention_low_rank_enabled=True,
        attention_rank=512,
        attention_q_nonlinearity=MoSENonlinearity.identity,
        attention_k_nonlinearity=MoSENonlinearity.identity,
        attention_v_nonlinearity=MoSENonlinearity.identity,
        attention_o_nonlinearity=MoSENonlinearity.identity,
    )
    config.block.feed_forward = replace(
        config.block.feed_forward,
        share_linear_gate_up_subspace=True,
        share_nonlinear_gate_up_subspace=False,
    )
    validate_model(config, target)
    return config


def validate_model(model, target):
    shared_rank, private_rank = VARIANTS[target]
    ff = model.block.feed_forward
    att = model.block.sequence_mixer
    if (model.d_model, ff.hidden_size) != (2048, 8192):
        raise ValueError("UG sweep requires the OLMo3-1B dimensions")
    if (ff.r1, ff.r2) != (shared_rank, private_rank):
        raise ValueError("rank overrides would change the registered UG variant")
    if (ff.share_linear_gate_up_subspace, ff.share_nonlinear_gate_up_subspace) != (
        True, False
    ):
        raise ValueError("UG sweep requires a shared r1 branch and private r2 branches")
    if ff.down_r1 != 0 or ff.down_r2 != 512:
        raise ValueError("Down rank must stay 512")
    if ff.down_nonlinearity != MoSENonlinearity.rms_norm or ff.rms_norm_learnable_weight:
        raise ValueError("Down must use RMSNorm without a learnable gain")
    if ff.gate_nonlinearity != MoSENonlinearity.identity or ff.up_nonlinearity != MoSENonlinearity.identity:
        raise ValueError("UG intermediate activations must be Identity")
    if ff.control != SwiGLUChannelControl.standard or ff.control_scope != SwiGLUChannelControlScope.none:
        raise ValueError("Keep the native SwiGLU gate")
    for projection in ("q", "k", "v", "o"):
        rank = getattr(att, f"{projection}_rank") or att.rank
        if rank != 512 or getattr(att, f"{projection}_nonlinearity") != MoSENonlinearity.identity:
            raise ValueError("All attention projections must remain independent linear rank-512")
    if str(att.share_scope) != "none":
        raise ValueError("Attention sharing is outside this UG sweep")
    # Exclude the unchanged Q/K norm gains from the projection-only budget.
    actual = ff.num_params(model.d_model) + 4 * (2 * model.d_model * 512)
    expected = REFERENCE_PROJECTION_PARAMS_PER_BLOCK + (-163_840 if target == "p1" else 0)
    if actual != expected:
        raise ValueError(f"Projection count mismatch: {actual:,} != {expected:,}")


def make_build_config(target):
    def build_config(opts, overrides):
        opts.sequence_length = 8192
        config = build_pretrain_config(
            opts, [], lambda tokenizer: build_model(tokenizer, target),
            variant=variant_name(target),
        )
        config.init_seed = 12536
        config.data_loader.seed = 34521
        config.data_loader.global_batch_size = 256 * 8192
        config.train_module.rank_microbatch_size = 32768
        config.train_module.optim.lr = 0.005
        config.train_module.scheduler = WSD(warmup=2000, decay_fraction=0.2)
        config.train_module.float8_config.enabled = True
        config.trainer.max_duration = Duration.tokens(30_000_000_000)
        config.trainer.callbacks["checkpointer"].save_interval = 1788
        config.trainer.callbacks["checkpointer"].max_checkpoints = None
        config = config.merge(overrides)
        validate_model(config.model, target)
        return config
    return build_config
