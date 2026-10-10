"""C0: K-only SiLU rank-512 control with Down RMSNorm."""

from _common import make_build_config
from olmo_mose import LowRankAttentionSharingScope

build_config = make_build_config(
    variant="r512matched-c0-k-silu-down-rms",
    share_gate_up_subspace=False,
    gate_up_rank=512,
    attention_share_scope=LowRankAttentionSharingScope.none,
    q_rank=512,
    k_rank=512,
    v_rank=512,
    o_rank=512,
    down_rank=512,
)


if __name__ == "__main__":
    from olmo_core.script_utils import main

    main(build_config)
