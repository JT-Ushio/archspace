"""Dense-budget single Q/K/Up/Gate substitutions reuse the projection adapter."""

from olmo_mose.dense_single_attention import DenseSingleSiLUAttention
from olmo_mose.dense_single_ffn import DenseSingleSiLUFFN
from ..registry import register
from .dense import attention, dense_ffn

register(DenseSingleSiLUAttention)(attention)
register(DenseSingleSiLUFFN)(dense_ffn)
