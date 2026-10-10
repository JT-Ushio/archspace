"""Selective Linear input factors plus RMS Down reuse the shared HF FFN runtime."""

from olmo_mose.dense_ffn_factorization import DenseFFNFactorization

from ..registry import register
from .dense import dense_ffn

register(DenseFFNFactorization)(dense_ffn)
