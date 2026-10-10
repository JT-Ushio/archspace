"""Compare every W1/W1-QK initial parameter using the actual CPU initializer.

This allocates two full FP32 models (about 6.4 GB of parameter storage). It does
not start training, load data, contact the scheduler, or inspect GPU checkpoints.
"""

import argparse
import importlib.util
import json
from argparse import Namespace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import torch

from olmo_core.utils import seed_all


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    backend = Path(__file__).resolve().parents[3]
    common_path = backend / "cfps/BS256_Seq8K_30B_WSD_FP8/FFN_Width_Ablations/_common.py"
    spec = importlib.util.spec_from_file_location("width_initialization_config", common_path)
    common = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(common)
    opts = Namespace(name="initialization-audit", sequence_length=8192, data_root="/tmp/data",
                     save_folder="/tmp/checkpoints", work_dir="/tmp/work")
    configs = [common.make_build_config(target)(opts, []) for target in ("w1", "w1qk")]
    models = []
    for target, config in zip(("w1", "w1qk"), configs):
        seed_all(config.init_seed)
        model = config.model.build(init_device="meta")
        model.init_weights(device=torch.device("cpu"))
        models.append(model)
        print(f"{target}: initialized full 16-layer model", flush=True)

    left, right = [dict(model.named_parameters()) for model in models]
    assert left.keys() == right.keys(), "parameter names differ"
    manifest = []
    for name, value in left.items():
        other = right[name]
        assert value.shape == other.shape and value.dtype == other.dtype, name
        assert torch.equal(value, other), f"initial parameter differs: {name}"
        # Check finite values in bounded chunks rather than allocate a full mask.
        for chunk in value.detach().reshape(-1).split(1_048_576):
            assert torch.isfinite(chunk).all(), f"non-finite initial parameter: {name}"
        manifest.append({"name": name, "shape": list(value.shape), "dtype": str(value.dtype),
                         "numel": value.numel(), "elementwise_equal": True})
    left_buffers, right_buffers = [dict(model.named_buffers()) for model in models]
    assert left_buffers.keys() == right_buffers.keys(), "buffer names differ"
    for name, value in left_buffers.items():
        assert torch.equal(value, right_buffers[name]), f"initial buffer differs: {name}"

    result = {
        "observed_at_hkt": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "variants": ["w1", "w1qk"], "device": "cpu", "layers": 16,
        "experiment_init_seed": configs[0].init_seed,
        "model_weight_init_seed": configs[0].model.init_seed,
        "data_seed": configs[0].data_loader.seed,
        "method": "torch.equal on every real full-model parameter and registered buffer",
        "all_parameters_elementwise_equal": True,
        "all_buffers_elementwise_equal": True,
        "parameter_tensor_count": len(manifest),
        "parameter_element_count": sum(row["numel"] for row in manifest),
        "buffer_tensor_count": len(left_buffers),
        "scope": "CPU FP32 weights before FP8/FSDP; actual GPU step0 checkpoints not checked",
        "parameters": manifest,
    }
    assert result["parameter_element_count"] == 797_050_880
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "parameters"},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
