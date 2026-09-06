import sys
from pathlib import Path

import pytest

from olmo_core.config import DType
from olmo_core.nn.attention import (
    AttentionConfig,
    GateGranularity,
)
from olmo_core.nn.layer_norm import LayerNormType
from olmo_core.nn.transformer.init import InitMethod
from olmo_ahn import LinearAttentionConfig

CFGS_DIR = Path(__file__).parents[1] / "cfgs"
sys.path.insert(0, str(CFGS_DIR))

from _qwen35_407m_models import (  # noqa: E402
    FFN_HIDDEN_SIZE,
    FULL_HEAD_DIM,
    FULL_N_HEADS,
    FULL_N_KV_HEADS,
    GDN_CONV_SIZE,
    GDN_HEAD_DIM,
    GDN_N_HEADS,
    GDN_N_V_HEADS,
    NORM_EPS,
    PARTIAL_ROTARY_FACTOR,
    ROPE_THETA,
    SWA_WINDOW_SIZE,
    TARGET_HYBRID_SCALING_PARAMS,
    TARGET_HYBRID_TOTAL_PARAMS,
    build_qwen35_407m_all_full,
    build_qwen35_407m_all_gdn,
    build_qwen35_407m_all_swa,
    build_qwen35_407m_gdn3_full1,
    build_qwen35_407m_swa3_full1,
)


class _Tokenizer:
    def padded_vocab_size(self) -> int:
        return 100_352


BUILDERS = {
    "all-gdn": build_qwen35_407m_all_gdn,
    "gdn3-full1": build_qwen35_407m_gdn3_full1,
    "all-full": build_qwen35_407m_all_full,
    "swa3-full1": build_qwen35_407m_swa3_full1,
    "all-swa": build_qwen35_407m_all_swa,
}


@pytest.mark.parametrize("builder", BUILDERS.values(), ids=BUILDERS.keys())
def test_qwen35_407m_shared_backbone_config(builder) -> None:
    config = builder(_Tokenizer())

    assert config.d_model == 1_024
    assert config.n_layers == 16
    assert config.vocab_size == 100_352
    assert config.init_method == InitMethod.scaled_truncated_normal
    assert config.tie_word_embeddings is False
    assert config.lm_head.bias is False
    assert config.lm_head.layer_norm is not None
    assert config.lm_head.layer_norm.name == LayerNormType.qwen_rms
    assert config.lm_head.layer_norm.eps == NORM_EPS
    for block in config.resolved_block_configs:
        assert block.feed_forward.hidden_size == FFN_HIDDEN_SIZE
        assert block.feed_forward.bias is False
        assert block.layer_norm.name == LayerNormType.qwen_rms
        assert block.layer_norm.eps == NORM_EPS
        assert block.layer_norm.bias is False


def test_qwen35_407m_full_attention_details() -> None:
    config = build_qwen35_407m_all_full(_Tokenizer())
    attention = config.block.sequence_mixer

    assert isinstance(attention, AttentionConfig)
    assert attention.n_heads == FULL_N_HEADS
    assert attention.n_kv_heads == FULL_N_KV_HEADS
    assert attention.head_dim == FULL_HEAD_DIM
    assert attention.dtype == DType.float32
    assert attention.bias is False
    assert attention.dropout == 0.0
    assert attention.gate is not None
    assert attention.gate.granularity == GateGranularity.elementwise
    assert attention.qk_norm is not None
    assert attention.qk_norm.name == LayerNormType.qwen_rms
    assert attention.qk_norm.eps == NORM_EPS
    assert attention.use_head_qk_norm is True
    assert attention.rope is not None
    assert attention.rope.theta == ROPE_THETA
    assert attention.rope.partial_rotary_factor == PARTIAL_ROTARY_FACTOR


def test_qwen35_407m_gdn_details() -> None:
    config = build_qwen35_407m_all_gdn(_Tokenizer())
    attention = config.block.sequence_mixer

    assert isinstance(attention, LinearAttentionConfig)
    assert attention.n_heads == GDN_N_HEADS
    assert attention.n_v_heads == GDN_N_V_HEADS
    assert attention.head_dim == GDN_HEAD_DIM
    assert attention.expand_v == 1.0
    assert attention.conv_size == GDN_CONV_SIZE
    assert attention.conv_bias is False
    assert attention.allow_neg_eigval is False
    assert attention.norm_eps == NORM_EPS
    assert attention.dtype == DType.float32
    assert attention._resolved_dimensions(1_024) == (16, 128, 128, 1.0)


def _linear_indices(config) -> list[int]:
    return [
        index
        for index, block in enumerate(config.resolved_block_configs)
        if isinstance(block.sequence_mixer, LinearAttentionConfig)
    ]


def _swa_indices(config) -> list[int]:
    indices = []
    for index, block in enumerate(config.resolved_block_configs):
        attention = block.sequence_mixer
        if isinstance(attention, AttentionConfig) and attention.sliding_window is not None:
            if attention.sliding_window.should_use_swa(index, config.n_layers):
                indices.append(index)
    return indices


def test_qwen35_407m_attention_layouts() -> None:
    expected_three_to_one = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]

    assert _linear_indices(build_qwen35_407m_all_gdn(_Tokenizer())) == list(range(16))
    assert _linear_indices(build_qwen35_407m_gdn3_full1(_Tokenizer())) == expected_three_to_one
    assert _swa_indices(build_qwen35_407m_swa3_full1(_Tokenizer())) == expected_three_to_one
    assert _swa_indices(build_qwen35_407m_all_swa(_Tokenizer())) == list(range(16))
    assert _swa_indices(build_qwen35_407m_all_full(_Tokenizer())) == []

    swa = build_qwen35_407m_all_swa(_Tokenizer()).block.sequence_mixer
    assert isinstance(swa, AttentionConfig)
    assert swa.sliding_window is not None
    assert swa.sliding_window.pattern == [SWA_WINDOW_SIZE]


def test_qwen35_407m_hybrid_parameter_targets() -> None:
    config = build_qwen35_407m_gdn3_full1(_Tokenizer())

    assert config.num_params == TARGET_HYBRID_TOTAL_PARAMS
    assert config.num_non_embedding_params == TARGET_HYBRID_SCALING_PARAMS
