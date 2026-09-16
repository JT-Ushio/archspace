"""Two local CPU workers verify artifact loading into genuinely sharded factors.

Run with torchrun --standalone --nproc-per-node=2 tests/svd_dtensor_smoke.py.
"""
import json
import shutil
import tempfile
from pathlib import Path
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, distribute_tensor
from test_svd_initialization import tiny, artifacts
from olmo_mose.svd_initialization import apply_artifacts, config_fingerprint, projection_pairs


def main():
    torch.set_num_threads(2)
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    directory = [tempfile.mkdtemp(prefix="mose-svd-dtensor-") if rank == 0 else None]
    dist.broadcast_object_list(directory)
    root = Path(directory[0])
    try:
        cfg = tiny()
        model = cfg.build(init_device="cpu")
        model.init_weights(device=torch.device("cpu"), max_seq_len=8)
        if rank == 0:
            artifacts(model, cfg, root)
        dist.barrier()
        mesh = init_device_mesh("cpu", (dist.get_world_size(),))
        for enc, dec in projection_pairs(model).values():
            for module in (enc, dec):
                module.weight = torch.nn.Parameter(distribute_tensor(module.weight.detach(), mesh, [Shard(0)]))
        apply_artifacts(model, root, "svd_scale_matched", config_fingerprint(cfg))
        manifest = json.loads((root / "manifest.json").read_text())
        for name, (enc, dec) in projection_pairs(model).items():
            saved = torch.load(root / manifest["pairs"][name]["file"], weights_only=True)["svd_scale_matched"]
            torch.testing.assert_close(enc.weight.full_tensor(), saved["E"], atol=0, rtol=0)
            torch.testing.assert_close(dec.weight.full_tensor(), saved["D"], atol=0, rtol=0)
        print(f"rank{rank}: DTENSOR ARTIFACT ROUNDTRIP PASSED", flush=True)
        dist.barrier()
    finally:
        dist.destroy_process_group()
        if rank == 0:
            # Only this test's newly allocated temporary fixture is removed.
            shutil.rmtree(root)


if __name__ == "__main__":
    main()
