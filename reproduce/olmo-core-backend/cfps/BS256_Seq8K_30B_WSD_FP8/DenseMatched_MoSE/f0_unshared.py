"""F0: dense-parameter-matched low-dimensional bottlenecks without sharing."""

from _common import make_build_config
from olmo_mose import LowRankAttentionSharingScope

build_config = make_build_config(
    variant="dense-matched-mose-f0-unshared-down-rms",
    share_gate_up_subspace=False,
    gate_up_rank=1640,
    attention_share_scope=LowRankAttentionSharingScope.none,
    attention_q_rank=1024,
    attention_k_rank=1024,
    attention_v_rank=1024,
    attention_o_rank=1024,
    down_rank=1632,
)


if __name__ == "__main__":
    from olmo_core.script_utils import main

    main(build_config)
