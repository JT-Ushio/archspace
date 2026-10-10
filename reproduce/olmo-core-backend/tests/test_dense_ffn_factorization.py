"""Selective Dense-budget FFN cases; HF numerical checks are opt-in."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from olmo_core.nn.feed_forward import FeedForwardConfig
from olmo_core.nn.transformer.init import InitMethod
from olmo_mose.attention import NonlinearLowRankProjection
from olmo_mose.dense_ffn_factorization import DenseFFNFactorizationConfig, TARGETS
from olmo_mose.feed_forward import MoSENonlinearity


@pytest.mark.parametrize("target", list(TARGETS))
def test_meta_structure_and_roundtrip(target):
    config = DenseFFNFactorizationConfig(
        hidden_size=32, bias=False, target=target, input_rank=6, down_rank=6
    )
    restored = FeedForwardConfig.from_dict(config.as_config_dict())
    assert type(restored) is DenseFFNFactorizationConfig
    module = restored.build(8, init_device="meta")
    assert sum(p.numel() for p in module.parameters()) == config.num_params(8)
    assert module.w2.nonlinearity == MoSENonlinearity.rms_norm
    assert module.w2.nonlinear_norm.weight is None
    for name in ("w1", "w3"):
        assert isinstance(getattr(module, name), NonlinearLowRankProjection) == (
            name in TARGETS[target]
        )
        if name in TARGETS[target]:
            assert getattr(module, name).nonlinearity == MoSENonlinearity.identity
    factors = [getattr(module, name).v for name in TARGETS[target]]
    assert len({id(x) for x in factors}) == len(factors)


@pytest.mark.parametrize("target", list(TARGETS))
def test_training_formula_and_gradients(target):
    module = DenseFFNFactorizationConfig(
        hidden_size=32, bias=False, target=target, input_rank=6, down_rank=6
    ).build(8)
    InitMethod.normal.init_feed_forward(module, d_model=8, block_idx=0, num_blocks=1)
    x = torch.randn(2, 3, 8, requires_grad=True)

    def input_projection(name):
        p = getattr(module, name)
        return p.u(p.v(x)) if name in TARGETS[target] else p(x)

    hidden = torch.nn.functional.silu(input_projection("w1")) * input_projection("w3")
    latent = module.w2.v(hidden)
    expected = module.w2.u(module.w2.nonlinear_norm(latent))
    output = module(x)
    torch.testing.assert_close(output, expected)
    output.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())


@pytest.mark.parametrize("target", list(TARGETS))
def test_registered_recipe_and_hf_structure(target):
    from olmo_mose.hf_export.audit import make_hf_config, structural_audit
    from olmo_mose.hf_export.checkpoint import build_native_meta

    root = Path(__file__).resolve().parents[1]
    p = root / "cfps/BS256_Seq8K_30B_WSD_FP8/Dense_FFN_Linear_RMS/_common.py"
    spec = importlib.util.spec_from_file_location("dense_ffn_common_test", p)
    common = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(common)
    opts = SimpleNamespace(
        name="dense-ffn-meta",
        data_root="/unused/data",
        work_dir="/tmp/meta",
        save_folder="/unused/ckpt",
        sequence_length=None,
    )
    config = common.make_build_config(target)(opts, [])
    assert config.init_seed == 12536 and config.data_loader.seed == 34521
    assert config.data_loader.global_batch_size == 2097152
    assert config.train_module.rank_microbatch_size == 32768
    assert config.trainer.max_duration.value == 30000000000
    assert config.train_module.float8_config.enabled
    native = build_native_meta(config.as_config_dict())
    audit = structural_audit(native, make_hf_config(native, config.as_config_dict()))
    assert audit["unique_parameters"] == common.expected_params(target)
