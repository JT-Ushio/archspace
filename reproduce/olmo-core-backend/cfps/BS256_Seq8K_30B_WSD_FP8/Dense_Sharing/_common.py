"""Dense baseline recipe with one dense-matched nonlinear attention projection."""
import sys
from pathlib import Path

from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import WSD
from olmo_core.train import Duration
from olmo_mose.dense_shared_attention import DenseSharedKVConfig
from olmo_mose import MoSESwiGLUConfig, MoSENonlinearity, SwiGLUChannelControl, SwiGLUChannelControlScope

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
    if target not in ('s1','s2','s3'):raise ValueError(target)
    for block in blocks:
        if target in ('s2','s3'):
            block.sequence_mixer = DenseSharedKVConfig(**block.sequence_mixer.as_dict(recurse=False),rank=1360)
        if target in ('s1','s3'):
            block.feed_forward = MoSESwiGLUConfig(**block.feed_forward.as_dict(recurse=False),
                r1=0,r2=1808,down_r1=0,down_r2=0,
                control=SwiGLUChannelControl.standard,control_scope=SwiGLUChannelControlScope.none,
                gate_nonlinearity=MoSENonlinearity.identity,up_nonlinearity=MoSENonlinearity.identity,
                share_gate_up_subspace=True)
    expected_delta={'s1':-3670016,'s2':-524288,'s3':-4194304}[target]
    assert config.num_params == dense_count + expected_delta
    return config


def make_build_config(target):
    def build_config(opts, overrides):
        opts.sequence_length = 8192
        config = build_pretrain_config(
            opts, [], lambda tokenizer: build_model(tokenizer, target),
            variant=f"dense-sharing-{target}-linear",
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
