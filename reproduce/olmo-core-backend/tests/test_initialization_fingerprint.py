"""Full-parameter hashing preserves state and handles empty DTensor shards."""

import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.tensor import Shard, distribute_tensor, init_device_mesh

from olmo_mose.initialization_fingerprint import InitializationFingerprintCallback


def record(model, folder, step=0):
    callback = InitializationFingerprintCallback()
    callback.trainer = SimpleNamespace(global_step=step, save_folder=str(folder),
                                       train_module=SimpleNamespace(model=model))
    callback.pre_train()


def test_fingerprint_preserves_weights_rng_and_detects_changes(tmp_path):
    model = torch.nn.Linear(3, 2, dtype=torch.bfloat16)
    before = {name: value.clone() for name, value in model.named_parameters()}
    rng = torch.random.get_rng_state().clone()
    record(model, tmp_path / "first")
    record(model, tmp_path / "same")
    first = json.loads((tmp_path / "first/initialization_fingerprint.json").read_text())
    same = json.loads((tmp_path / "same/initialization_fingerprint.json").read_text())
    assert first == same
    assert first["parameter_elements"] == 8
    assert torch.equal(rng, torch.random.get_rng_state())
    assert all(torch.equal(before[name], value) for name, value in model.named_parameters())
    with torch.no_grad():
        model.weight[0, 0] += 1
    record(model, tmp_path / "changed")
    changed = json.loads((tmp_path / "changed/initialization_fingerprint.json").read_text())
    assert first["parameter_sha256"] != changed["parameter_sha256"]
    record(model, tmp_path / "resume", step=1788)
    assert not (tmp_path / "resume").exists()


def worker(rank, rendezvous, folder):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    mesh = init_device_mesh("cpu", (2,))
    model = torch.nn.Module()
    model.register_parameter("singleton", torch.nn.Parameter(
        distribute_tensor(torch.tensor([1.25]), mesh, [Shard(0)])
    ))
    if rank == 1:
        assert model.singleton.to_local().numel() == 0
    record(model, folder)
    dist.barrier()
    dist.destroy_process_group()


def test_two_rank_full_hash_with_empty_shard(tmp_path):
    mp.spawn(worker, args=(str(tmp_path / "rendezvous"), str(tmp_path / "distributed")),
             nprocs=2, join=True)
    model = torch.nn.Module()
    model.register_parameter("singleton", torch.nn.Parameter(torch.tensor([1.25])))
    record(model, tmp_path / "local")
    def read(folder):
        return json.loads((Path(folder) / "initialization_fingerprint.json").read_text())
    assert read(tmp_path / "local") == read(tmp_path / "distributed")
