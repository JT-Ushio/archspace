"""Meta-only budget, saved-config and shared HF adapter checks; no forward."""

import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
BACKEND = HERE.parents[2]
CONFIG_ROOT = BACKEND / "cfps/BS256_Seq8K_30B_WSD_FP8/FFN_Residual_Bottlenecks"
sys.path[:0] = [str(CONFIG_ROOT), str(BACKEND / "src")]

from _common import TOTAL_PARAMS, VARIANTS, variant_name  # noqa: E402
from olmo_mose.hf_export.audit import make_hf_config, structural_audit  # noqa: E402
from olmo_mose.hf_export.checkpoint import build_native_meta  # noqa: E402


def main():
    results = []
    compact_states = []
    for target, (width, rank, wide, latent) in VARIANTS.items():
        path = CONFIG_ROOT / f"{target}.py"
        spec = importlib.util.spec_from_file_location(f"residual_cfp_{target}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        opts = SimpleNamespace(
            name=f"olmo3-1b-{variant_name(target)}-fp8-30b",
            data_root="/unused/data", work_dir="/tmp/mose-residual-meta",
            save_folder="/unused/ckpt", sequence_length=None,
        )
        config = module.build_config(opts, [])
        raw = config.as_config_dict()
        native = build_native_meta(raw)
        count = sum(p.numel() for p in native.parameters())
        assert count == TOTAL_PARAMS, (target, count)
        assert config.init_seed == 12536 and config.model.init_seed == 0
        assert config.data_loader.seed == 34521
        assert config.data_loader.global_batch_size == 2097152
        assert config.train_module.rank_microbatch_size == 32768
        assert config.train_module.optim.lr == 0.005
        assert config.train_module.scheduler.warmup == 2000
        assert config.train_module.scheduler.decay_fraction == 0.2
        assert config.train_module.float8_config.enabled is True
        assert config.trainer.max_duration.value == 30000000000
        assert config.trainer.callbacks["checkpointer"].save_interval == 1788
        assert config.trainer.callbacks["checkpointer"].max_checkpoints is None
        audit = structural_audit(native, make_hf_config(native, raw))
        signature = {k: list(v.shape) for k, v in native.state_dict().items()}
        if target.startswith("r"):
            compact_states.append(signature)
        results.append({
            "variant": target, "width": width, "rank": rank,
            "wide_residual": wide, "latent_residual": latent,
            "run_name": opts.name, "parameters": count,
            "model": raw["model"], "hf_structural_audit": audit,
            "numerical_hf_validation": "not_run",
            "seeds": {"model": 0, "runtime": 12536, "data": 34521},
            "recipe": {"global_batch_tokens": 2097152, "sequence_length": 8192,
                       "microbatch_tokens": 32768, "accumulation_at_8gpu": 8,
                       "tokens": 30000000000, "muon_lr": 0.005,
                       "warmup_steps": 2000, "decay_fraction": 0.2},
        })
    assert all(s == compact_states[0] for s in compact_states)
    report = {
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "meta_only": True, "numerical_hf_validation": "not_run",
        "r0_r3_state_keys_and_shapes_identical": True,
        "gpu_initialization_fingerprints": "pending_actual_training",
        "configs": results,
    }
    (HERE / "CONFIG_AUDIT.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps({
        "parameters": {row["variant"]: row["parameters"] for row in results},
        "hf_structural_check": "passed", "numerical_hf_validation": "not_run",
        "r0_r3_state_keys_and_shapes_identical": True,
    }))


if __name__ == "__main__":
    main()
