"""Dense Attention; selected FFN input factors Linear and Down factors RMSNorm."""

import sys
from pathlib import Path

from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import WSD
from olmo_core.train import Duration
from olmo_mose.dense_ffn_factorization import DenseFFNFactorizationConfig, TARGETS

CFGS_DIR = Path(__file__).resolve().parents[3] / "cfgs"
if str(CFGS_DIR) not in sys.path:
    sys.path.insert(0, str(CFGS_DIR))
from _pretrain_common import build_pretrain_config  # noqa: E402

RANK = 1632
DENSE_MODEL_PARAMS = 1_484_916_736


def variant_name(target):
    return f"dense-nearmatched-{target.replace('_', '-')}-linear-r{RANK}-down-rms-r{RANK}"


def expected_params(target):
    return DENSE_MODEL_PARAMS - 16 * (len(TARGETS[target]) + 1) * 65536


def build_model(tokenizer, target):
    model = TransformerConfig.olmo3_1B(
        vocab_size=tokenizer.padded_vocab_size(),
        attn_backend=AttentionBackendName.flash_3,
    )
    dense_count = model.num_params
    blocks = list(model.block.values()) if isinstance(model.block, dict) else [model.block]
    if model.block_overrides:
        blocks += list(model.block_overrides.values())
    for block in blocks:
        block.feed_forward = DenseFFNFactorizationConfig(
            **block.feed_forward.as_dict(recurse=False),
            target=target,
            input_rank=RANK,
            down_rank=RANK,
        )
    if dense_count != DENSE_MODEL_PARAMS or model.num_params != expected_params(target):
        raise ValueError("unexpected Dense reference or near-matched parameter budget")
    return model


def validate_model(model, target):
    ff = model.block.feed_forward
    if type(ff) is not DenseFFNFactorizationConfig or (
        model.d_model,
        model.n_layers,
        ff.hidden_size,
        ff.target,
        ff.input_rank,
        ff.down_rank,
        model.num_params,
    ) != (2048, 16, 8192, target, RANK, RANK, expected_params(target)):
        raise ValueError("the registered Dense FFN factorization configuration was changed")
    from olmo_core.nn.attention import AttentionConfig

    if type(model.block.sequence_mixer) is not AttentionConfig:
        raise ValueError("all Attention projections must remain native Dense")


def make_build_config(target):
    def build_config(opts, overrides):
        opts.sequence_length = 8192
        config = build_pretrain_config(
            opts,
            [],
            lambda tokenizer: build_model(tokenizer, target),
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
