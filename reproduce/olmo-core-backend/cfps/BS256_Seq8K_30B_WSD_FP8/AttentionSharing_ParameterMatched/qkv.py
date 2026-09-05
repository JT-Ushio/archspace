"""Parameter-matched Q/K/V-sharing ablation."""

from _common import make_build_config
from olmo_mose import LowRankAttentionSharingScope

build_config = make_build_config(
    variant="parameter-matched-qkv",
    share_gate_up_subspace=False,
    up_gate_rank=512,
    attention_share_scope=LowRankAttentionSharingScope.qkv,
    attention_q_rank=768,
    attention_k_rank=768,
    attention_v_rank=768,
    attention_o_rank=512,
)


if __name__ == "__main__":
    from olmo_core.script_utils import main

    main(build_config)
