"""Dense baseline recipe with one dense-matched nonlinear attention projection."""
import sys
from pathlib import Path

from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import WSD
from olmo_core.train import Duration
from olmo_mose.dense_single_attention import DenseSingleSiLUAttentionConfig

CFGS_DIR = Path(__file__).resolve().parents[3] / "cfgs"
if str(CFGS_DIR) not in sys.path:
    sys.path.insert(0, str(CFGS_DIR))
from _pretrain_common import build_pretrain_config


def build_model(tokenizer, target):
    config = TransformerConfig.olmo3_1B(
        vocab_size=tokenizer.padded_vocab_size(),
        attn_backend=AttentionBackendName.flash_3,
    )
    dense_count = config.num_params
    blocks = list(config.block.values()) if isinstance(config.block, dict) else [config.block]
    if config.block_overrides:
        blocks += list(config.block_overrides.values())
    for block in blocks:
        att = block.sequence_mixer
        block.sequence_mixer = DenseSingleSiLUAttentionConfig(
            **att.as_dict(recurse=False), target=target)
    assert config.num_params == dense_count, "dense parameter budget changed"
    return config


def make_build_config(target):
    def build_config(opts, overrides):
        opts.sequence_length = 8192
        config = build_pretrain_config(
            opts, [], lambda tokenizer: build_model(tokenizer, target),
            variant=f"dense-matched-{target}-only-silu-r1024",
        )
        config.data_loader.global_batch_size = 256 * 8192
        config.train_module.rank_microbatch_size = 32768
        config.train_module.optim.lr = 0.005
        config.train_module.scheduler = WSD(warmup=2000, decay_fraction=0.2)
        config.train_module.float8_config.enabled = True
        config.trainer.max_duration = Duration.tokens(30_000_000_000)
        config.trainer.callbacks["checkpointer"].save_interval = 1788
        config.trainer.callbacks["checkpointer"].max_checkpoints = None
        return config.merge(overrides)
    return build_config
