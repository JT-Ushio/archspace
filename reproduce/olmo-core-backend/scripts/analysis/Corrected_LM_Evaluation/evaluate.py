"""Evaluate a completed checkpoint without entering a training loop."""

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import torch

from olmo_core.distributed.utils import get_rank, get_world_size
from olmo_core.train import prepare_training_environment, teardown_training_environment
from olmo_core.train.callbacks import CheckpointerCallback
from olmo_core.utils import seed_all

from alignment import DualMaskLMEvaluator, align_eval_batch
from case_evidence import validate_case

HERE = Path(__file__).resolve().parent
BACKEND = HERE.parents[2]


def reject_write(*args, **kwargs):
    raise RuntimeError("Standalone evaluation forbids training and checkpoint saves")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--manifest", type=Path, default=HERE / "cases.json")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    args = parser.parse_args()
    entry = json.loads(args.manifest.read_text())[args.case]
    checkpoint = validate_case(entry)
    config_path = BACKEND / entry["config_path"]
    sys.path.insert(0, str(config_path.parent))
    spec = importlib.util.spec_from_file_location("standalone_checkpoint_recipe", config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    opts = argparse.Namespace(name="corrected-lm-" + args.case, sequence_length=8192,
                              data_root=args.data_root,
                              save_folder=str(args.runtime_dir / "unused-checkpoints"),
                              work_dir=str(args.runtime_dir / "dataset-cache"))
    config = module.build_config(opts, [])
    if config.model.num_params != entry["num_params"]:
        raise ValueError("Reconstructed model parameter count differs from source")
    saved_config = json.loads((Path(checkpoint) / "config.json").read_text())
    if (saved_config["model"] != config.model.as_config_dict()
            or saved_config["init_seed"] != config.init_seed):
        raise ValueError("Reconstructed model/seed differs from exact checkpoint metadata")
    lm_config = config.trainer.callbacks["lm_evaluator"]
    # Use the same evaluation batch partition for the older Dense recipe too.
    config.train_module.rank_microbatch_size = 32768
    config.trainer.callbacks = {"lm_evaluator": lm_config,
                                "checkpointer": CheckpointerCallback(enabled=False)}
    config.trainer.async_bookkeeping = False
    config.trainer.save_overwrite = False
    config.trainer.work_dir = str(args.runtime_dir / "trainer")
    prepare_training_environment(shared_filesystem=True, backend="cpu:gloo,cuda:nccl")
    trainer = None
    try:
        if get_world_size() != 8:
            raise ValueError("Use one8GPU node for all standalone comparisons")
        seed_all(config.init_seed)
        model = config.model.build(init_device="meta")
        train_module = config.train_module.build(model)
        train_module.train_batch = reject_write
        train_module.optim_step = reject_write
        align_eval_batch(train_module)
        dataset = config.dataset.build()
        data_loader = config.data_loader.build(dataset, dp_process_group=train_module.dp_process_group)
        trainer = config.trainer.build(train_module, data_loader)
        trainer.save_checkpoint = reject_write
        trainer.save_checkpoint_async = reject_write
        trainer.load_checkpoint(checkpoint, load_trainer_state=False, load_optim_state=False)
        if not trainer.checkpoint_loaded:
            raise RuntimeError("Checkpoint was not loaded")
        callback = trainer.callbacks["lm_evaluator"]
        callback.evaluators = [DualMaskLMEvaluator(evaluator) for evaluator in callback.evaluators]
        callback.perform_eval(prefix="eval_mask_comparison")
        rows = {}
        for evaluator in callback.evaluators:
            metrics = evaluator.compute_metrics()
            for label in sorted(evaluator.metrics):
                rows[label] = {
                    "corrected_ce": float(metrics[f"{label}/CE loss"].item()),
                    "legacy_ce_same_forward": float(metrics[f"legacy/{label}/CE loss"].item()),
                    "corrected_scored_tokens": int(metrics[f"{label}/scored_tokens"].item()),
                    "legacy_selected_tokens": int(metrics[f"legacy/{label}/scored_tokens"].item()),
                }
        if len(rows) != 11 or any(row["corrected_scored_tokens"] <= 0 or
                                 not math.isfinite(row["corrected_ce"]) or
                                 not math.isfinite(row["legacy_ce_same_forward"]) for row in rows.values()):
            raise ValueError("Require finite scores with positive valid counts for all11 sources")
        result = {"case": args.case, "source": entry, "world_size": 8, "checkpoint_step": 14306,
                  "rows": rows, "corrected_macro_ce": sum(row["corrected_ce"] for row in rows.values()) / 11,
                  "legacy_macro_ce_same_forward": sum(row["legacy_ce_same_forward"] for row in rows.values()) / 11,
                  "scope": "Final checkpoint, same forward/data for legacy and corrected masks; no training",
                  "comparison_to_old_step14000_metrics": "Different checkpoint step; do not attribute that delta only to masks"}
        encoded = json.dumps(result, ensure_ascii=False, allow_nan=False)
        if get_rank() == 0:
            args.runtime_dir.mkdir(exist_ok=True, parents=True)
            (args.runtime_dir / "results.json").write_text(encoded)
            prefix = "MOSE_CORRECTED_LM_" + args.case.upper()
            print(prefix + "_SHA256 " + hashlib.sha256(encoded.encode()).hexdigest(), flush=True)
            print(prefix + "_JSON " + encoded, flush=True)
    finally:
        if trainer is not None:
            trainer._shutdown()
        teardown_training_environment()


if __name__ == "__main__":
    main()
