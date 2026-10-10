"""C3: combine K-only SiLU, Down RMSNorm, K/V and Gate/Up sharing."""

from _common import make_build_config
from olmo_mose import LowRankAttentionSharingScope

build_config = make_build_config(
    variant="r512matched-c3-k-silu-down-rms-kv-gateup",
    share_gate_up_subspace=True,
    gate_up_rank=568,
    attention_share_scope=LowRankAttentionSharingScope.kv,
    q_rank=520,
    k_rank=680,
    v_rank=680,
    o_rank=512,
    down_rank=512,
)


if __name__ == "__main__":
    from olmo_core.script_utils import main

    main(build_config)
