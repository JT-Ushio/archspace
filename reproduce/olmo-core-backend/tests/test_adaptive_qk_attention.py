"""Verify C2-equivalent initialization, learnability and checkpoint behavior."""

from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, distribute_tensor

from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.nn.transformer.init import InitMethod
from olmo_mose import LowRankAttentionConfig, MoSENonlinearity, SerializableMuonConfig
from olmo_mose.adaptive_qk_attention import (
    AdaptiveQKAttentionConfig,
    QKAlphaMetricsCallback,
)


def configs():
    kwargs = dict(
        n_heads=2, rank=4, bias=False, backend=AttentionBackendName.torch,
        q_nonlinearity=MoSENonlinearity.identity,
        k_nonlinearity=MoSENonlinearity.silu,
        v_nonlinearity=MoSENonlinearity.identity,
        o_nonlinearity=MoSENonlinearity.identity,
    )
    return LowRankAttentionConfig(**kwargs), AdaptiveQKAttentionConfig(**kwargs)


def initialize(module):
    module.init_weights(
        init_method=InitMethod.normal, d_model=8, block_idx=0, num_blocks=1,
        generator=torch.Generator().manual_seed(123),
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_initial_function_and_input_gradient_match_c2(dtype):
    base, adaptive = configs()
    ref = base.build(8, layer_idx=0, n_layers=1).to(dtype=dtype)
    model = adaptive.build(8, layer_idx=0, n_layers=1).to(dtype=dtype)
    initialize(ref)
    initialize(model)
    for name, parameter in ref.named_parameters():
        torch.testing.assert_close(parameter, dict(model.named_parameters())[name], rtol=0, atol=0)
    x = torch.randn(2, 5, 8, dtype=dtype, requires_grad=True)
    clone = x.detach().clone().requires_grad_(True)
    y, expected = model(x), ref(clone)
    torch.testing.assert_close(y, expected, rtol=0, atol=0)
    y.float().square().sum().backward()
    expected.float().square().sum().backward()
    torch.testing.assert_close(x.grad, clone.grad, rtol=0, atol=0)
    for gate in (model.w_q, model.w_k):
        assert gate.alpha.grad is not None and torch.isfinite(gate.alpha.grad).all()


def test_materialized_meta_gates_learn_and_roundtrip_checkpoint():
    _, config = configs()
    model = config.build(8, layer_idx=0, n_layers=1, init_device="meta").to_empty(device="cpu")
    initialize(model)
    assert model.w_q.alpha.item() == 0 and model.w_k.alpha.item() == 1
    before = [p.detach().clone() for p in (model.w_q.alpha, model.w_k.alpha)]
    optim = torch.optim.AdamW([model.w_q.alpha, model.w_k.alpha], lr=0.005, weight_decay=0)
    model(torch.randn(2, 6, 8)).square().sum().backward()
    optim.step()
    assert all(not torch.equal(a, b) for a, b in zip(before, (model.w_q.alpha, model.w_k.alpha)))
    restored = config.build(8, layer_idx=0, n_layers=1)
    restored.load_state_dict(model.state_dict())
    x = torch.randn(2, 6, 8)
    torch.testing.assert_close(model(x), restored(x), rtol=0, atol=0)
    merged = config.merge(["q_alpha_init=0.0", "k_alpha_init=1.0"])
    assert isinstance(merged, AdaptiveQKAttentionConfig)
    assert merged.num_params(8) == config.num_params(8)


def test_scalar_gates_are_adamw_without_decay_and_logged_by_layer():
    _, attention = configs()
    config = TransformerConfig.olmo3_1M(vocab_size=128, attn_backend=AttentionBackendName.torch)
    config.block.sequence_mixer = attention
    model = config.build(init_device="meta")
    groups = SerializableMuonConfig().build_groups(model)
    gates = [p for n, p in model.named_parameters() if n.endswith(".alpha")]
    for p in gates:
        group = next(g for g in groups if any(member is p for member in g["params"]))
        assert group["algorithm"] == "adamw" and group["weight_decay"] == 0
    assert gates
    # Log materialized values; the distributed path uses full_tensor on all ranks.
    model.to_empty(device="cpu")
    for name, parameter in model.named_parameters():
        if name.endswith(".alpha"):
            with torch.no_grad():
                parameter.fill_(0 if ".w_q." in name else 1)
    metrics = {}
    trainer = SimpleNamespace(
        global_step=100, train_module=SimpleNamespace(model=model),
        record_metric=lambda name, value: metrics.update({name: value.item()}),
    )
    callback = QKAlphaMetricsCallback(100)
    callback.trainer = trainer
    callback.post_step()
    assert metrics["qk_alpha/q_mean"] == 0 and metrics["qk_alpha/k_mean"] == 1
    assert len([name for name in metrics if name.startswith("qk_alpha/layer")]) == len(gates)


def test_gate_projection_can_be_captured_without_a_graph_break():
    _, config = configs()
    module = config.build(8, layer_idx=0, n_layers=1)
    initialize(module)
    x = torch.randn(2, 5, 8)
    compiled = torch.compile(module.w_q, backend="aot_eager", fullgraph=True)
    torch.testing.assert_close(compiled(x), module.w_q(x))


def _sharded_alpha_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    try:
        mesh = init_device_mesh("cpu", (2,))
        model = nn.Module()
        model.blocks = nn.ModuleDict({"0": nn.Module()})
        model.blocks["0"].attention = nn.Module()
        for name, value in [("w_q", 0.0), ("w_k", 1.0)]:
            projection = nn.Module()
            projection.alpha = nn.Parameter(
                distribute_tensor(torch.tensor([value]), mesh, [Shard(0)])
            )
            setattr(model.blocks["0"].attention, name, projection)
            assert projection.alpha.to_local().numel() == (1 if rank == 0 else 0)
        metrics = {}
        callback = QKAlphaMetricsCallback(100)
        callback.trainer = SimpleNamespace(
            global_step=100, train_module=SimpleNamespace(model=model),
            record_metric=lambda name, value: metrics.update({name: value.item()}),
        )
        callback.post_step()
        assert metrics["qk_alpha/layer0/q"] == metrics["qk_alpha/q_mean"] == 0
        assert metrics["qk_alpha/layer0/k"] == metrics["qk_alpha/k_mean"] == 1
    finally:
        dist.destroy_process_group()


def test_alpha_logging_restores_empty_shards_on_every_rank(tmp_path):
    mp.spawn(_sharded_alpha_worker, args=(str(tmp_path / "rendezvous"),), nprocs=2, join=True)
