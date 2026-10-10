"""C9: unshared rank512, Q-only SiLU, Down RMSNorm, other factors Linear."""

from _common import make_build_config
from olmo_mose import LowRankAttentionSharingScope, MoSENonlinearity

build_config = make_build_config(
    variant="r512matched-c9-q-silu-down-rms",
    share_gate_up_subspace=False,
    gate_up_rank=512,
    attention_share_scope=LowRankAttentionSharingScope.none,
    q_rank=512,
    k_rank=512,
    v_rank=512,
    o_rank=512,
    down_rank=512,
    q_nonlinearity=MoSENonlinearity.silu,
    k_nonlinearity=MoSENonlinearity.identity,
)


if __name__ == "__main__":
    from olmo_core.script_utils import main

    main(build_config)
