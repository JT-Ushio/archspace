"""C7: C6 with SiLU between the shared Gate/Up low-rank factors."""

from _common import make_build_config
from olmo_mose import LowRankAttentionSharingScope, MoSENonlinearity

build_config = make_build_config(
    variant="r512matched-c7-qk-silu-ug-silu-down-rms-gateup",
    share_gate_up_subspace=True,
    gate_up_rank=568,
    attention_share_scope=LowRankAttentionSharingScope.none,
    q_rank=512,
    k_rank=512,
    v_rank=512,
    o_rank=496,
    down_rank=520,
    q_nonlinearity=MoSENonlinearity.silu,
    k_nonlinearity=MoSENonlinearity.silu,
    gate_nonlinearity=MoSENonlinearity.silu,
    up_nonlinearity=MoSENonlinearity.silu,
)


if __name__ == "__main__":
    from olmo_core.script_utils import main

    main(build_config)
