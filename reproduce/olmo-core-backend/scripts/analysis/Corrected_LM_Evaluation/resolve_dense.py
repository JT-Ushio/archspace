"""Resolve the original Dense checkpoint from both exact resumed W&B segments."""

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

from resolve_references import merge_config, verify_checkpoint

HERE = Path(__file__).resolve().parent
BACKEND = HERE.parents[2]
RUN_ID = "olmo3-1b-dense-baseline-bs256-seq8k-30b-wsd-fp8-5e-3"
WANDB_ROOT = Path("/mnt/shared-storage-user/shared-storage-ailab-llmfudan/taoji/wandb/archspace/MoSE/wandb")
SEGMENTS = [WANDB_ROOT / ("offline-run-" + stamp + "-" + RUN_ID)
            for stamp in ("20260902_120045", "20260902_162134")]
CONFIG = "cfps/BS256_Seq8K_30B_WSD_FP8/OLMo3-1B-DenseBaseline.py"


def main():
    import wandb
    from wandb.proto import wandb_internal_pb2
    from wandb.sdk.internal.datastore import DataStore
    from wandb.sdk.lib import proto_util

    sys.path.insert(0, str(BACKEND / "scripts/train/UG_Partial_Sharing"))
    from compare_offline import compare

    wandb._assert_is_internal_process = True
    config, binaries, ids = {}, [], set()
    for segment in SEGMENTS:
        paths = list(segment.glob("run-*.wandb"))
        if len(paths) != 1:
            raise ValueError("Require one binary in each exact original Dense segment")
        binaries.append(str(paths[0]))
        store = DataStore()
        store.open_for_scan(binaries[-1])
        while (data := store.scan_data()) is not None:
            record = wandb_internal_pb2.Record.FromString(data)
            kind = record.WhichOneof("record_type")
            if kind == "run":
                ids.add(record.run.run_id)
                merge_config(config, proto_util.dict_from_proto_list(record.run.config.update))
            elif kind == "config":
                if record.config.remove:
                    raise ValueError("Resolve removed configuration fields explicitly")
                merge_config(config, proto_util.dict_from_proto_list(record.config.update))
    if ids != {RUN_ID}:
        raise ValueError("Original Dense segments do not identify the same expected run")
    history = compare({"dense": binaries}, 14306)
    if not history["full_30b_histories_complete"] or history["common_excluded_steps"]:
        raise ValueError("Original Dense history must contain all14306 positive finite CE values")
    checkpoint = verify_checkpoint(config, RUN_ID)
    saved = json.loads((Path(checkpoint) / "config.json").read_text())
    spec = importlib.util.spec_from_file_location("dense_checkpoint_recipe", BACKEND / CONFIG)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    opts = argparse.Namespace(name="resolve-original-dense", sequence_length=8192,
                              data_root="/unused", save_folder="/tmp/unused-save",
                              work_dir="/tmp/unused-work")
    recipe = module.build_config(opts, [])
    if (recipe.model.num_params != 1_484_916_736
            or recipe.model.as_config_dict() != saved["model"]
            or recipe.init_seed != saved["init_seed"]):
        raise ValueError("Dense recipe differs from the actual final checkpoint model/seed")
    result = {"source_job_id": None, "source_run_id": RUN_ID,
              "source_segments": [str(s) for s in SEGMENTS], "checkpoint": checkpoint,
              "config_path": CONFIG, "num_params": recipe.model.num_params,
              "init_seed": saved["init_seed"], "checkpoint_step": 14306,
              "training_steps": history["runs"]["dense"]["rows_in_window"],
              "train_ce": history["runs"]["dense"],
              "basis": "Actual W&B save_folder, full resumed history and final model/seed metadata; original scheduler job ID unavailable"}
    encoded = json.dumps(result, ensure_ascii=False, allow_nan=False)
    print("MOSE_DENSE_PATH_SHA256 " + hashlib.sha256(encoded.encode()).hexdigest(), flush=True)
    print("MOSE_DENSE_PATH_JSON " + encoded, flush=True)


if __name__ == "__main__":
    main()
