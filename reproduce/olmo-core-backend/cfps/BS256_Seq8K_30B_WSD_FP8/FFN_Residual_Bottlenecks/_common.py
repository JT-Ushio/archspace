"""Independent UG: original-width latent skip and compact 2x2 residual sweep."""

import sys
from pathlib import Path

from olmo_core.optim import WSD
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
from olmo_mose.initialization_fingerprint import InitializationFingerprintCallbackConfig  # noqa: E402
from olmo_mose.residual_feed_forward import ResidualMoSESwiGLUConfig  # noqa: E402

# width, rank, wide skip, Up-to-Down latent skip.
VARIANTS = {
    "l1": (8192, 512, False, True),
    "r0": (2048, 1280, False, False),
    "r1": (2048, 1280, True, False),
    "r2": (2048, 1280, False, True),
    "r3": (2048, 1280, True, True),
}
FFN_PARAMS_PER_LAYER = 15_728_640
TOTAL_PARAMS = 797_050_880


def variant_name(target):
    width, rank, _, _ = VARIANTS[target]
    return f"ffn-residual-{target}-m{width}-r{rank}-unshared-linear-down-rms"


def build_model(tokenizer, target):
    width, rank, wide, latent = VARIANTS[target]
    model = build_mose_olmo3_1b(
        tokenizer, SwiGLUChannelControl.standard,
        control_scope=SwiGLUChannelControlScope.none,
        r1=0, r2=rank, down_r1=0, down_r2=rank,
        share_gate_up_subspace=False,
        gate_nonlinearity=MoSENonlinearity.identity,
        up_nonlinearity=MoSENonlinearity.identity,
        down_nonlinearity=MoSENonlinearity.rms_norm,
        rms_norm_learnable_weight=False,
        attention_low_rank_enabled=True,
        attention_rank=512, attention_q_rank=512, attention_k_rank=512,
        attention_v_rank=512, attention_o_rank=512,
        attention_share_scope=LowRankAttentionSharingScope.none,
        attention_q_nonlinearity=MoSENonlinearity.identity,
        attention_k_nonlinearity=MoSENonlinearity.identity,
        attention_v_nonlinearity=MoSENonlinearity.identity,
        attention_o_nonlinearity=MoSENonlinearity.identity,
        attention_rms_norm_learnable_weight=False,
    )
    options = model.block.feed_forward.as_dict(exclude_none=True, recurse=False)
    options["hidden_size"] = width
    model.block.feed_forward = ResidualMoSESwiGLUConfig(
        **options, wide_residual=wide, latent_residual=latent,
    )
    model.init_seed = 0
    validate_model(model, target)
    return model


def validate_model(model, target):
    width, rank, wide, latent = VARIANTS[target]
    ff, att = model.block.feed_forward, model.block.sequence_mixer
    if (model.d_model, model.n_layers) != (2048, 16):
        raise ValueError("requires the existing OLMo3-1B model")
    if type(ff) is not ResidualMoSESwiGLUConfig:
        raise ValueError("residual config must retain its exact serialized class")
    if (ff.hidden_size, ff.r1, ff.r2, ff.down_r1, ff.down_r2) != (width, 0, rank, 0, rank):
        raise ValueError("FFN width/ranks differ from the registered ablation")
    if (ff.wide_residual, ff.latent_residual) != (wide, latent):
        raise ValueError("residual switches differ from the registered ablation")
    ff.__post_init__()
    if ff.num_params(model.d_model) != FFN_PARAMS_PER_LAYER:
        raise ValueError("FFN projection budget mismatch")
    if att.bias or att.rms_norm_learnable_weight or str(att.share_scope) != "none":
        raise ValueError("Attention must be independent and bias-free")
    for name in ("q", "k", "v", "o"):
        if (getattr(att, f"{name}_rank") or att.rank) != 512 or (
            getattr(att, f"{name}_nonlinearity") != MoSENonlinearity.identity
        ):
            raise ValueError("all Attention factors must be rank512 and Linear")


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
        checkpointer = config.trainer.callbacks["checkpointer"]
        checkpointer.save_interval = 1788
        checkpointer.max_checkpoints = None
        config.trainer.callbacks["initialization_fingerprint"] = (
            InitializationFingerprintCallbackConfig()
        )
        config = config.merge(overrides)
        validate_model(config.model, target)
        return config
    return build_config
