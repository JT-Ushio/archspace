"""C2-based width/bottleneck tradeoffs and a separate optional Q/K gate."""

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
from olmo_mose.adaptive_qk_attention import (  # noqa: E402
    AdaptiveQKAttentionConfig,
    QKAlphaMetricsCallbackConfig,
)
from olmo_mose.initialization_fingerprint import (  # noqa: E402
    InitializationFingerprintCallbackConfig,
)

VARIANTS = {
    "w1": (6144, 736, 640),
    "w1qk": (6144, 736, 640),
    "w1m": (6144, 568, 520),
    "w2": (4096, 976, 944),
    "g1": (8192, 568, 520),
}
FFN_PARAMS_PER_LAYER = 15_794_176
REFERENCE_MODEL_PARAMS = 797_050_880


def expected_ffn_params(target):
    return 12_402_688 if target == "w1m" else FFN_PARAMS_PER_LAYER


def expected_model_params(target):
    return REFERENCE_MODEL_PARAMS + 16 * (
        expected_ffn_params(target) - FFN_PARAMS_PER_LAYER
    ) + (32 if target == "g1" else 0)


def variant_name(target):
    width, ug, down = VARIANTS[target]
    if target == "g1":
        return "qk-adaptive-g1-c2-q0-k1-m8192-ug568-down520-down-rms"
    if target == "w1qk":
        return f"ffn-width-{target}-m{width}-ug{ug}-down{down}-c6-qk-silu-down-rms"
    return f"ffn-width-{target}-m{width}-ug{ug}-down{down}-c2-k-silu-down-rms"


def build_model(tokenizer, target):
    width, ug, down = VARIANTS[target]
    model = build_mose_olmo3_1b(
        tokenizer, SwiGLUChannelControl.standard,
        control_scope=SwiGLUChannelControlScope.none,
        r1=0, r2=ug, down_r1=0, down_r2=down,
        share_gate_up_subspace=True,
        gate_nonlinearity=MoSENonlinearity.identity,
        up_nonlinearity=MoSENonlinearity.identity,
        down_nonlinearity=MoSENonlinearity.rms_norm,
        rms_norm_learnable_weight=False,
        attention_low_rank_enabled=True,
        attention_rank=512, attention_q_rank=512, attention_k_rank=512,
        attention_v_rank=512, attention_o_rank=496,
        attention_share_scope=LowRankAttentionSharingScope.none,
        attention_q_nonlinearity=(MoSENonlinearity.silu if target == "w1qk"
                                  else MoSENonlinearity.identity),
        attention_k_nonlinearity=MoSENonlinearity.silu,
        attention_v_nonlinearity=MoSENonlinearity.identity,
        attention_o_nonlinearity=MoSENonlinearity.identity,
        attention_rms_norm_learnable_weight=False,
    )
    model.block.feed_forward.hidden_size = width
    if target == "g1":
        model.block.sequence_mixer = AdaptiveQKAttentionConfig(
            **model.block.sequence_mixer.as_dict(exclude_none=True, recurse=False),
            q_alpha_init=0.0, k_alpha_init=1.0,
        )
    validate_model(model, target)
    return model


def validate_model(model, target):
    width, ug, down = VARIANTS[target]
    ff, att = model.block.feed_forward, model.block.sequence_mixer
    if (model.d_model, model.n_layers) != (2048, 16):
        raise ValueError("this experiment requires OLMo3-1B dimensions")
    if (ff.hidden_size, ff.r1, ff.r2, ff.down_r1, ff.down_r2) != (width, 0, ug, 0, down):
        raise ValueError("width or bottleneck ranks differ from the registered experiment")
    if not ff.share_gate_up_subspace or any(v is not None for v in (
        ff.share_linear_gate_up_subspace, ff.share_nonlinear_gate_up_subspace,
    )):
        raise ValueError("UG must use complete input-factor sharing")
    if ff.bias or ff.rms_norm_learnable_weight or (
        ff.gate_nonlinearity != MoSENonlinearity.identity
        or ff.up_nonlinearity != MoSENonlinearity.identity
        or ff.down_nonlinearity != MoSENonlinearity.rms_norm
        or ff.control != SwiGLUChannelControl.standard
        or ff.control_scope != SwiGLUChannelControlScope.none
    ):
        raise ValueError("keep native SwiGLU, UG Linear and no-gain Down RMSNorm")
    if att.bias or att.rms_norm_learnable_weight or str(att.share_scope) != "none":
        raise ValueError("Attention must retain independent bias-free C2 projections")
    for projection, rank, activation in (
        ("q", 512, MoSENonlinearity.silu if target == "w1qk" else MoSENonlinearity.identity),
        ("k", 512, MoSENonlinearity.silu),
        ("v", 512, MoSENonlinearity.identity),
        ("o", 496, MoSENonlinearity.identity),
    ):
        if (getattr(att, f"{projection}_rank") or att.rank) != rank or (
            getattr(att, f"{projection}_nonlinearity") != activation
        ):
            raise ValueError("Attention ranks and activations differ from the registered experiment")
    gated = isinstance(att, AdaptiveQKAttentionConfig)
    if gated != (target == "g1") or (gated and (att.q_alpha_init, att.k_alpha_init) != (0, 1)):
        raise ValueError("G1 alone enables adaptive Q/K, initialized to C2 endpoints")
    if ff.num_params(model.d_model) != expected_ffn_params(target):
        raise ValueError("FFN parameter budget differs from the registered experiment")
    if model.num_params != expected_model_params(target):
        raise ValueError("full-model parameter budget differs from the registered experiment")


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
        if target == "g1":
            config.trainer.callbacks["qk_alpha"] = QKAlphaMetricsCallbackConfig(interval=100)
        if target in ("w1", "w1qk"):
            config.trainer.callbacks["initialization_fingerprint"] = (
                InitializationFingerprintCallbackConfig()
            )
        config = config.merge(overrides)
        validate_model(config.model, target)
        return config
    return build_config
