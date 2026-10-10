"""C4: all-Attention SiLU + FFN Identity/Down RMSNorm + K/V sharing."""

from _common import make_build_config
from olmo_mose import LowRankAttentionSharingScope, MoSENonlinearity

build_config = make_build_config(
    variant="r512matched-c4-attention-silu-down-rms-kv",
    share_gate_up_subspace=False,
    gate_up_rank=512,
    attention_share_scope=LowRankAttentionSharingScope.kv,
    q_rank=512,
    k_rank=688,
    v_rank=688,
    o_rank=504,
    down_rank=512,
    q_nonlinearity=MoSENonlinearity.silu,
    k_nonlinearity=MoSENonlinearity.silu,
    v_nonlinearity=MoSENonlinearity.silu,
    o_nonlinearity=MoSENonlinearity.silu,
)


if __name__ == "__main__":
    from olmo_core.script_utils import main

    main(build_config)
