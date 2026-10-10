"""Allocate more of the fixed rank512 projection budget to Attention."""

import sys
from pathlib import Path

from olmo_core.optim import WSD
from olmo_core.train import Duration

CFGS_DIR = Path(__file__).resolve().parents[3] / "cfgs"
if str(CFGS_DIR) not in sys.path:
    sys.path.insert(0, str(CFGS_DIR))

from _models import build_mose_olmo3_1b  # noqa: E402
from _pretrain_common import build_pretrain_config  # noqa: E402
from olmo_mose import MoSENonlinearity, SwiGLUChannelControl, SwiGLUChannelControlScope  # noqa: E402

VARIANTS = {"a1": (336, 352, False), "a2": (320, 448, True)}


def variant_name(target):
    ug_rank, down_rank, shared = VARIANTS[target]
    scope = "shared" if shared else "unshared"
    return f"attn-budget-{target}-r832-ug{ug_rank}-down{down_rank}-{scope}-qk-silu-down-rms"


def validate_model(model, target):
    ug_rank, down_rank, shared = VARIANTS[target]
    ff = model.block.feed_forward
    att = model.block.sequence_mixer
    if (model.d_model, model.n_layers, ff.hidden_size) != (2048, 16, 8192):
        raise ValueError("Attention reallocation requires OLMo3-1B dimensions")
    if (ff.r1, ff.r2, ff.down_r1, ff.down_r2, ff.share_gate_up_subspace) != (
        0, ug_rank, 0, down_rank, shared
    ):
        raise ValueError("FFN overrides would change the registered rank allocation")
    if ff.share_linear_gate_up_subspace is not None or ff.share_nonlinear_gate_up_subspace is not None:
        raise ValueError("Use uniform UG sharing")
    if ff.gate_nonlinearity != MoSENonlinearity.identity or ff.up_nonlinearity != MoSENonlinearity.identity:
        raise ValueError("UG intermediate activations must be Identity")
    if ff.down_nonlinearity != MoSENonlinearity.rms_norm or ff.rms_norm_learnable_weight:
        raise ValueError("Down must use RMSNorm without a learnable gain")
    if ff.control != SwiGLUChannelControl.standard or ff.control_scope != SwiGLUChannelControlScope.none:
        raise ValueError("Keep the native SwiGLU gate")
    if str(att.share_scope) != "none":
        raise ValueError("Attention remains independent")
    for projection in ("q", "k", "v", "o"):
        if (getattr(att, f"{projection}_rank") or att.rank) != 832:
            raise ValueError("All attention ranks must remain832")
        expected = MoSENonlinearity.silu if projection in ("q", "k") else MoSENonlinearity.identity
        if getattr(att, f"{projection}_nonlinearity") != expected:
            raise ValueError("Attention requires Q/K SiLU and V/O Identity")
    if ff.num_params(model.d_model) != 10_485_760:
        raise ValueError("FFN must use the registered10,485,760 parameter budget")
    if att.num_params(model.d_model) != 13_635_584:
        raise ValueError("Attention budget must include rank832 projections and native QK norms")
    if model.num_params != 797_050_880:
        raise ValueError("Full-model budget must exactly match independent rank512")


def build_model(tokenizer, target):
    ug_rank, down_rank, shared = VARIANTS[target]
    config = build_mose_olmo3_1b(
        tokenizer, SwiGLUChannelControl.standard,
        control_scope=SwiGLUChannelControlScope.none,
        r1=0, r2=ug_rank, down_r1=0, down_r2=down_rank,
        gate_nonlinearity=MoSENonlinearity.identity,
        up_nonlinearity=MoSENonlinearity.identity,
        down_nonlinearity=MoSENonlinearity.rms_norm,
        rms_norm_learnable_weight=False,
        share_gate_up_subspace=shared,
        attention_low_rank_enabled=True, attention_rank=832,
        attention_q_nonlinearity=MoSENonlinearity.silu,
        attention_k_nonlinearity=MoSENonlinearity.silu,
        attention_v_nonlinearity=MoSENonlinearity.identity,
        attention_o_nonlinearity=MoSENonlinearity.identity,
    )
    validate_model(config, target)
    return config


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
