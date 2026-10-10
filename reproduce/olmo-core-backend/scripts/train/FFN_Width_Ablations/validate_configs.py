"""Audit real 16-layer meta-model counts without loading data or training."""

import argparse
import importlib.util
import json
from argparse import Namespace
from datetime import datetime, timedelta, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("w1", "w1qk", "w1m", "w2", "g1"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    backend = Path(__file__).resolve().parents[3]
    common_path = backend / "cfps/BS256_Seq8K_30B_WSD_FP8/FFN_Width_Ablations/_common.py"
    spec = importlib.util.spec_from_file_location("width_validation_config", common_path)
    common = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(common)
    opts = Namespace(name="parameter-audit", sequence_length=8192, data_root="/tmp/data",
                     save_folder="/tmp/checkpoints", work_dir="/tmp/work")
    rows = []
    for target in ([args.variant] if args.variant else common.VARIANTS):
        config = common.make_build_config(target)(opts, [])
        model = config.model.build(init_device="meta")
        actual = sum(p.numel() for p in model.parameters())
        assert actual == config.model.num_params
        ff = model.blocks["0"].feed_forward
        width, ug, down = common.VARIANTS[target]
        formula = (2048 + 2 * width) * ug + (width + 2048) * down
        assert formula == sum(p.numel() for p in ff.parameters()) == common.expected_ffn_params(target)
        assert actual == common.expected_model_params(target)
        gate_count = sum(p.numel() for n, p in model.named_parameters() if n.endswith(".alpha"))
        assert gate_count == (32 if target == "g1" else 0)
        rows.append({
            "variant": target, "status": "prepared_not_submitted",
            "ffn_width": width, "ug_shared_rank": ug, "down_rank": down,
            "ffn_parameters_per_layer": formula, "actual_model_parameters": actual,
            "extra_gate_parameters": gate_count, "strict_budget_match": actual == 797_050_880,
            "parameter_delta_from_c2": actual - 797_050_880,
            "budget_role": "width_only_smaller_control" if target == "w1m" else (
                "adaptive_gate_backup" if target == "g1" else "matched_budget_main"
            ),
            "q_nonlinearity": str(config.model.block.sequence_mixer.q_nonlinearity),
            "k_nonlinearity": str(config.model.block.sequence_mixer.k_nonlinearity),
            "local_jacobian_rank_upper_bound": min(ug, down),
            "attention_ranks": {"q": 512, "k": 512, "v": 512, "o": 496},
            "global_batch_tokens": config.data_loader.global_batch_size,
            "rank_microbatch_tokens": config.train_module.rank_microbatch_size,
            "supported_gpu_counts": [n for n in (8, 16, 32) if not (
                config.data_loader.global_batch_size % (config.train_module.rank_microbatch_size * n)
            )],
            "init_seed": config.init_seed, "model_weight_init_seed": config.model.init_seed,
            "data_seed": config.data_loader.seed,
            "max_tokens": config.trainer.max_duration.value,
            "muon_lr": config.train_module.optim.lr,
            "warmup_steps": config.train_module.scheduler.warmup,
            "decay_fraction": config.train_module.scheduler.decay_fraction,
            "fp8": config.train_module.float8_config.enabled,
            "checkpoint_interval": config.trainer.callbacks["checkpointer"].save_interval,
            "keep_all_checkpoints": config.trainer.callbacks["checkpointer"].max_checkpoints is None,
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "observed_at_hkt": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "method": "Real full-model and per-layer parameter tensors on the meta device",
        "scope": "Architecture/recipe validation; not a distributed FP8 training run",
        "runs": rows,
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
