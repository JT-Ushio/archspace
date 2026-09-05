"""Parameter-matched up/gate-sharing ablation."""

from _common import make_build_config
from olmo_mose import LowRankAttentionSharingScope

build_config = make_build_config(
    variant="parameter-matched-up-gate",
    share_gate_up_subspace=True,
    up_gate_rank=568,
    attention_share_scope=LowRankAttentionSharingScope.none,
    attention_q_rank=512,
    attention_k_rank=512,
    attention_v_rank=512,
    attention_o_rank=512,
)


if __name__ == "__main__":
    from olmo_core.script_utils import main

    main(build_config)
