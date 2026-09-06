import pytest
import torch
from torch import nn
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_optimizer_state_dict,
    set_optimizer_state_dict,
)

from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.transformer import Transformer, TransformerConfig
from olmo_ahn import SerializableMuonConfig


class _StateOnlyOptimizer(torch.optim.Optimizer):
    """Use PyTorch checkpoint handling without importing Dion's CUDA kernels."""

    def __init__(self, params, distributed_mesh=None, **defaults):
        defaults.setdefault("step", 0)
        super().__init__(params, defaults)
        for group in self.param_groups:
            for param in group["params"]:
                self.state[param]["momentum"] = torch.zeros_like(param)


def _build_model(*, n_kv_heads: int, device: str) -> Transformer:
    model = TransformerConfig.llama_like(
        d_model=64,
        vocab_size=128,
        n_layers=1,
        n_heads=8,
        n_kv_heads=n_kv_heads,
        head_dim=8,
        attn_backend=AttentionBackendName.torch,
    ).build(init_device=device)
    # Exercise the same 3D depthwise-weight routing as GDN without requiring FLA.
    model.blocks["0"].conv_probe = nn.Conv1d(
        64, 64, kernel_size=4, groups=64, bias=False, device=device
    )
    return model


@pytest.mark.parametrize("n_kv_heads", [1, 2])
def test_muon_preserves_per_head_groups_and_routes_convolution_to_adamw(n_kv_heads: int) -> None:
    model = _build_model(n_kv_heads=n_kv_heads, device="meta")
    config = SerializableMuonConfig()
    groups = config.build_groups(model)
    groups_by_param = {id(param): group for group in groups for param in group["params"]}

    assert config.flatten is False
    assert sum(len(group["params"]) for group in groups) == len(groups_by_param)
    assert set(groups_by_param) == {id(param) for param in model.parameters()}
    for projection, heads in (("w_q", 8), ("w_k", n_kv_heads), ("w_v", n_kv_heads)):
        param = model.get_parameter(f"blocks.0.attention.{projection}.weight")
        group = groups_by_param[id(param)]
        assert group.get("algorithm", "muon") == "muon"
        assert group.get("num_heads", 1) == heads
        assert group.get("flatten", config.flatten) is False

    conv = model.get_parameter("blocks.0.conv_probe.weight")
    assert conv.ndim == 3
    assert groups_by_param[id(conv)]["algorithm"] == "adamw"

    # Building groups must not pollute the config saved for a later run.
    restored = SerializableMuonConfig.from_dict(config.as_config_dict())
    assert restored.group_overrides is None
    assert restored.flatten is False
    assert len(restored.build_groups(model)) == len(groups)


@pytest.mark.parametrize("checkpoint_format", ["torch", "dcp", "dcp_flattened"])
def test_checkpoint_cannot_restore_legacy_flatten_setting(monkeypatch, checkpoint_format) -> None:
    monkeypatch.setattr(
        SerializableMuonConfig, "optimizer", classmethod(lambda cls: _StateOnlyOptimizer)
    )
    model = _build_model(n_kv_heads=1, device="cpu")
    legacy = SerializableMuonConfig(flatten=True, lr=1e-3).build(model)
    for group in legacy.param_groups:
        group["step"] = 7
    for param_state in legacy.state.values():
        param_state["momentum"].fill_(0.25)

    restored = SerializableMuonConfig(lr=2e-3).build(model)
    if checkpoint_format == "torch":
        restored.load_state_dict(legacy.state_dict())
    else:
        options = StateDictOptions(flatten_optimizer_state_dict=checkpoint_format == "dcp_flattened")
        state = get_optimizer_state_dict(model, legacy, options=options)
        set_optimizer_state_dict(model, restored, state, options=options)

    assert any(group.get("num_heads", 1) > 1 for group in restored.param_groups)
    for group in restored.param_groups:
        assert group["flatten"] is False
        assert group["step"] == 7
        assert group["lr"] == 1e-3  # The scheduler's checkpoint LR is still restored.
        assert group["initial_lr"] == 2e-3  # Existing fixed-field behavior is preserved.
        for param in group["params"]:
            torch.testing.assert_close(
                restored.state[param]["momentum"], torch.full_like(param, 0.25)
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/Dion")
@pytest.mark.parametrize("resume_legacy_checkpoint", [False, True])
def test_muon_cuda_first_step_with_per_head_and_convolution_groups(
    resume_legacy_checkpoint: bool,
) -> None:
    pytest.importorskip("dion")
    model = _build_model(n_kv_heads=1, device="cuda")
    config = SerializableMuonConfig(lr=2e-3, use_polar_express=False)
    with torch.no_grad():
        for param in model.parameters():
            param.normal_(std=0.02)
            param.grad = torch.randn_like(param)

    optim = config.build(model)
    if resume_legacy_checkpoint:
        legacy_state = optim.state_dict()
        for group in legacy_state["param_groups"]:
            group["flatten"] = True
        optim.load_state_dict(legacy_state)
    q = model.get_parameter("blocks.0.attention.w_q.weight")
    conv = model.get_parameter("blocks.0.conv_probe.weight")
    q_before, conv_before = q.detach().clone(), conv.detach().clone()
    optim.step()
    torch.cuda.synchronize()

    assert all(torch.isfinite(param).all() for param in model.parameters())
    assert not torch.equal(q, q_before)
    assert not torch.equal(conv, conv_before)
