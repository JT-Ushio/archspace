import sys
from pathlib import Path

CFGS_DIR = Path(__file__).parents[1] / "cfgs"
sys.path.insert(0, str(CFGS_DIR))

from _models import build_mla  # noqa: E402
from olmo_mla_morphnorm import MLAAttentionConfig, MLANormType  # noqa: E402


class _Tokenizer:
    def padded_vocab_size(self) -> int:
        return 100_352


def test_mla_model_uses_separate_qk_and_value_head_dimensions() -> None:
    config = build_mla(_Tokenizer(), MLANormType.baseline)
    attention = config.block.sequence_mixer

    assert isinstance(attention, MLAAttentionConfig)
    assert attention._resolved_dims(config.d_model) == (192, 128, 16, 128)
