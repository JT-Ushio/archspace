"""C5: Q-only SiLU + Down RMSNorm + parameter-reinvested Gate/Up sharing."""

from _common import make_build_config
from olmo_mose import LowRankAttentionSharingScope, MoSENonlinearity

build_config = make_build_config(
    variant="r512matched-c5-q-silu-down-rms-gateup",
    share_gate_up_subspace=True,
    gate_up_rank=568,
    attention_share_scope=LowRankAttentionSharingScope.none,
    q_rank=512,
    k_rank=512,
    v_rank=512,
    o_rank=496,
    down_rank=520,
    q_nonlinearity=MoSENonlinearity.silu,
    k_nonlinearity=MoSENonlinearity.identity,
)


if __name__ == "__main__":
    from olmo_core.script_utils import main

    main(build_config)
