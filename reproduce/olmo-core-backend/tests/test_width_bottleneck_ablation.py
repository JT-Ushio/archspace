"""Budget accounting, unchanged recipe, and actual changed FFN computation."""

import importlib.util
from argparse import Namespace
from pathlib import Path

import pytest
import torch

from olmo_core.data import TokenizerConfig
from olmo_core.nn.transformer.init import InitMethod

BACKEND = Path(__file__).resolve().parents[1]


def load_common(family, module_name):
    path = BACKEND / f"cfps/BS256_Seq8K_30B_WSD_FP8/{family}/_common.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("target", ["w1", "w1qk", "w1m", "w2", "g1"])
def test_full_model_actual_counts_and_unchanged_c2_recipe(target):
    common = load_common("FFN_Width_Ablations", "width_test_config")
    reference = load_common("Rank512Matched_Combinations", "width_reference_config")
    opts = Namespace(name="audit", sequence_length=8192, data_root="/tmp/data",
                     save_folder="/tmp/checkpoints", work_dir="/tmp/work")
    c2 = reference.make_build_config(
        variant="reference", share_gate_up_subspace=True, gate_up_rank=568,
        attention_share_scope=reference.LowRankAttentionSharingScope.none,
        q_rank=512, k_rank=512, v_rank=512, o_rank=496, down_rank=520,
    )(opts, [])
    actual = common.make_build_config(target)(opts, [])
    c2.model = actual.model
    c2.trainer.callbacks["wandb"].group = common.variant_name(target)
    if target == "g1":
        c2.trainer.callbacks["qk_alpha"] = actual.trainer.callbacks["qk_alpha"]
    if target in ("w1", "w1qk"):
        c2.trainer.callbacks["initialization_fingerprint"] = actual.trainer.callbacks[
            "initialization_fingerprint"
        ]
    assert c2.as_dict() == actual.as_dict()
    expected = 742_787_072 if target == "w1m" else 797_050_880 + (32 if target == "g1" else 0)
    assert actual.model.num_params == expected
    model = actual.model.build(init_device="meta")
    assert sum(p.numel() for p in model.parameters()) == expected
    ff = model.blocks["0"].feed_forward
    assert sum(p.numel() for p in ff.parameters()) == (
        12_402_688 if target == "w1m" else 15_794_176
    )
    assert actual.model.block.sequence_mixer.q_nonlinearity == (
        common.MoSENonlinearity.silu if target == "w1qk" else common.MoSENonlinearity.identity
    )
    assert actual.model.block.sequence_mixer.k_nonlinearity == common.MoSENonlinearity.silu
    alphas = [p for name, p in model.named_parameters() if name.endswith(".alpha")]
    assert len(alphas) == (32 if target == "g1" else 0)
    assert all(p.shape == (1,) for p in alphas)
    for world_size in (8, 16, 32):
        assert actual.data_loader.global_batch_size % (
            actual.train_module.rank_microbatch_size * world_size
        ) == 0
    actual.model.block.feed_forward.hidden_size += 16
    with pytest.raises(ValueError, match="width or bottleneck"):
        common.validate_model(actual.model, target)


@pytest.mark.parametrize("target", ["w1", "w1m", "w2"])
def test_actual_width_ffn_forward_backward(target):
    common = load_common("FFN_Width_Ablations", "width_forward_config")
    config = common.build_model(TokenizerConfig.dolma2(), target)
    ff = config.block.feed_forward.build(2048, init_device="cpu")
    InitMethod.normal.init_feed_forward(ff, d_model=2048, block_idx=0, num_blocks=16)
    x = torch.randn(1, 2, 2048, requires_grad=True)
    y = ff(x)
    assert y.shape == x.shape
    y.square().mean().backward()
    assert torch.isfinite(y).all() and torch.isfinite(x.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in ff.parameters())


def test_w1_qk_changes_only_q_activation_and_no_parameter_shapes():
    common = load_common("FFN_Width_Ablations", "width_qk_pair_config")
    w1 = common.build_model(TokenizerConfig.dolma2(), "w1")
    w1qk = common.build_model(TokenizerConfig.dolma2(), "w1qk")
    assert w1.block.sequence_mixer.q_nonlinearity == common.MoSENonlinearity.identity
    assert w1qk.block.sequence_mixer.q_nonlinearity == common.MoSENonlinearity.silu
    w1qk.block.sequence_mixer.q_nonlinearity = common.MoSENonlinearity.identity
    assert w1.as_dict() == w1qk.as_dict()
