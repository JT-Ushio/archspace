"""Matched QK-Identity/SiLU comparisons across partial sharing ratios."""
import importlib.util
from dataclasses import replace
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "ug_partial_primary_recipe", Path(__file__).with_name("_common.py")
)
primary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(primary)
MoSENonlinearity = primary.MoSENonlinearity
VARIANTS = {"p0qk": "p0", "p2qk": "p2", "p3qk": "p3", "p4qk": "p4"}


def variant_name(target):
    shared, private = primary.VARIANTS[VARIANTS[target]]
    return f"ug-partial-{target}-s{shared}-p{private}-linear-qk-silu-down-rms"


def validate_model(model, target):
    att = model.block.sequence_mixer
    if (att.q_nonlinearity, att.k_nonlinearity) != (
        MoSENonlinearity.silu, MoSENonlinearity.silu
    ):
        raise ValueError("This interaction experiment requires Q and K SiLU")
    # Reuse primary guards on a copy with Q/K reset. Never modify the model.
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
    if model.num_params != 797_050_880:
        raise ValueError("Full-model budget must exactly match independent rank512")


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
