"""Paired initialization-seed replications of the completed C0/C2/C6 recipes."""

import importlib.util
from pathlib import Path

path = Path(__file__).resolve().parents[1] / "Rank512Matched_Combinations/_common.py"
spec = importlib.util.spec_from_file_location("rank512_original_recipe", path)
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)

VARIANTS = {"c0s2": ("c0", 12537), "c6s2": ("c6", 12537),
            "c0s3": ("c0", 12538), "c6s3": ("c6", 12538),
            "c2s2": ("c2", 12537), "c2s3": ("c2", 12538)}
DATA_SEED = 34521


def variant_name(target):
    base, seed = VARIANTS[target]
    return f"r512matched-{base}-seed{seed}-data{DATA_SEED}"


def reference_builder(base):
    shared = base in ("c2", "c6")
    names = {"c0": "r512matched-c0-k-silu-down-rms",
             "c2": "r512matched-c2-k-silu-down-rms-gateup",
             "c6": "r512matched-c6-qk-silu-down-rms-gateup"}
    return reference.make_build_config(
        variant=names[base],
        share_gate_up_subspace=shared,
        gate_up_rank=568 if shared else 512,
        attention_share_scope=reference.LowRankAttentionSharingScope.none,
        q_rank=512, k_rank=512, v_rank=512,
        o_rank=496 if shared else 512,
        down_rank=520 if shared else 512,
        q_nonlinearity=(reference.MoSENonlinearity.silu if base == "c6" else
                        reference.MoSENonlinearity.identity),
        k_nonlinearity=reference.MoSENonlinearity.silu,
    )


def make_build_config(target):
    base, seed = VARIANTS[target]

    def build_config(opts, overrides):
        config = reference_builder(base)(opts, [])
        original_model = config.model.as_dict()
        config.init_seed = seed
        config.data_loader.seed = DATA_SEED
        config.trainer.callbacks["wandb"].group = variant_name(target)
        config = config.merge(overrides)
        reference._validate_rank512_budget(config)
        if config.model.as_dict() != original_model:
            raise ValueError("Seed replication must retain the original C0/C2/C6 architecture")
        if (config.init_seed, config.data_loader.seed) != (seed, DATA_SEED):
            raise ValueError("Seed replication requires its registered initialization/data seeds")
        return config

    return build_config
