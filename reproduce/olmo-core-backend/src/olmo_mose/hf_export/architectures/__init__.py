"""Import registrations; adding an architecture adds a module here, not a new CLI."""

from . import (  # noqa: F401
    adaptive_qk,
    channel_control,
    dense,
    dense_ffn_factorization,
    dense_shared_kv,
    dense_single_silu,
    initialization,
    low_rank_attention,
    mose_swiglu,
    output_rotation,
    residual_feed_forward,
)
