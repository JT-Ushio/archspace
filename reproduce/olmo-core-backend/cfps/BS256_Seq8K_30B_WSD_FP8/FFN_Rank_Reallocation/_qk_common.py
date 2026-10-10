"""Combine Q/K SiLU with fixed-budget FFN rank allocations."""

import importlib.util
from dataclasses import replace
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "ffn_reallocation_primary_recipe", Path(__file__).with_name("_common.py")
)
primary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(primary)
MoSENonlinearity = primary.MoSENonlinearity
VARIANTS = {"f1qk": "f1", "f2qk": "f2", "f3qk": "f3", "f4qk": "f4", "f5qk": "f5"}


def variant_name(target):
    ug_rank, down_rank, shared = primary.VARIANTS[VARIANTS[target]]
    scope = "shared" if shared else "unshared"
    return f"ffn-budget-{target}-ug{ug_rank}-down{down_rank}-{scope}-qk-silu-ug-linear-down-rms"


def validate_model(model, target):
    att = model.block.sequence_mixer
    if (att.q_nonlinearity, att.k_nonlinearity) != (
        MoSENonlinearity.silu, MoSENonlinearity.silu
    ):
        raise ValueError("This combination requires Q and K SiLU")
    reference = replace(
        model,
        block=replace(
            model.block,
            sequence_mixer=replace(
                att, q_nonlinearity=MoSENonlinearity.identity,
                k_nonlinearity=MoSENonlinearity.identity,
            ),
        ),
    )
    primary.validate_model(reference, VARIANTS[target])


def make_build_config(target):
    def build_config(opts, overrides):
        config = primary.make_build_config(VARIANTS[target])(opts, [])
        att = config.model.block.sequence_mixer
        att.q_nonlinearity = MoSENonlinearity.silu
        att.k_nonlinearity = MoSENonlinearity.silu
        config.trainer.callbacks["wandb"].group = variant_name(target)
        config = config.merge(overrides)
        validate_model(config.model, target)
        return config
    return build_config
