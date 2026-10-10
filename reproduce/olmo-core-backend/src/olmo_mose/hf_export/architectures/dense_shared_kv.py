"""Dense Q/O and a shared linear K/V bottleneck."""

from olmo_mose.dense_shared_attention import DenseSharedKVAttention
from ..registry import register
from .low_rank_attention import shared_attention

register(DenseSharedKVAttention)(shared_attention)
