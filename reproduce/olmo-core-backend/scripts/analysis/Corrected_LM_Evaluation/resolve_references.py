"""Resolve archived final checkpoints from their actual W&B training configs.

Run inside a scheduled allocation: the W&B/checkpoint mount is unavailable on
the local development machine. This does not infer roots from today's launcher.
"""

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REFERENCES = HERE.parents[1] / "train/UG_Partial_Sharing/historical_reference_segments.json"


def merge_config(current, updates):
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(current.get(key), dict):
            merge_config(current[key], value)
        else:
            current[key] = value


def verify_checkpoint(config, run_id):
    folder = config["trainer"]["save_folder"]
    if not isinstance(folder, str) or not folder.endswith("/" + run_id + "/stage1"):
        raise ValueError("Saved training folder does not identify this run and stage")
    checkpoint = Path(folder) / "step14306"
    saved = json.loads((checkpoint / "config.json").read_text())
    for key in ("model", "init_seed"):
        if saved[key] != config[key]:
            raise ValueError("Final checkpoint metadata differs from the training config")
    return str(checkpoint)


def main():
    import wandb
    from wandb.proto import wandb_internal_pb2
    from wandb.sdk.internal.datastore import DataStore
    from wandb.sdk.lib import proto_util

    wandb._assert_is_internal_process = True
    references = json.loads(REFERENCES.read_text())
    if set(references) != {"c0-original", "c2-original", "c6-original"}:
        raise ValueError("Resolve only the three archived references")
    results = {}
    for case, source in references.items():
        ledger = Path(source["completion_ledger"]).read_text()
        if source["job_id"] not in ledger:
            raise ValueError("Reference source differs from the archived ledger")
        paths = list(Path(source["segment"]).glob("*.wandb"))
        if len(paths) != 1:
            raise ValueError("Require one binary in the exact archived segment")
        config, steps, ids = {}, set(), set()
        store = DataStore()
        store.open_for_scan(str(paths[0]))
        while (data := store.scan_data()) is not None:
            record = wandb_internal_pb2.Record.FromString(data)
            kind = record.WhichOneof("record_type")
            if kind == "run":
                ids.add(record.run.run_id)
                merge_config(config, proto_util.dict_from_proto_list(record.run.config.update))
            elif kind == "config":
                if record.config.remove:
                    raise ValueError("Resolve removed config fields explicitly")
                merge_config(config, proto_util.dict_from_proto_list(record.config.update))
            elif kind == "history":
                values = proto_util.dict_from_proto_list(record.history.item)
                if "train/CE loss" in values:
                    steps.add(int(values.get("_step", record.history.step.num)))
        if len(ids) != 1 or steps != set(range(1, 14307)):
            raise ValueError("Require a unique run and its complete 30B history")
        run_id = ids.pop()
        if not Path(source["segment"]).name.endswith("-" + run_id):
            raise ValueError("Binary run ID differs from the exact archived segment")
        checkpoint = verify_checkpoint(config, run_id)
        results[case] = {"source": source, "checkpoint": checkpoint,
                         "run_id": run_id, "init_seed": config["init_seed"],
                         "training_steps": len(steps), "checkpoint_step": 14306,
                         "basis": "Actual offline training save_folder and matching final config.json; archived completion ledger"}
    encoded = json.dumps(results, ensure_ascii=False, allow_nan=False)
    print("MOSE_REFERENCE_PATHS_SHA256 " + hashlib.sha256(encoded.encode()).hexdigest(), flush=True)
    print("MOSE_REFERENCE_PATHS_JSON " + encoded, flush=True)


if __name__ == "__main__":
    main()
