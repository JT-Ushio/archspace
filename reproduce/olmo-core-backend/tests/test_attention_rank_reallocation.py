"""Verify real module budgets and unchanged training recipe for Attention sweeps."""

import importlib.util
from argparse import Namespace
from pathlib import Path

import pytest


def load_config_module(family, module_name):
    path = Path(__file__).resolve().parents[1] / f"cfps/BS256_Seq8K_30B_WSD_FP8/{family}/_common.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_budget_and_recipe_with_real_modules():
    common = load_config_module("Attention_Rank_Reallocation", "attention_budget_test")
    reference_module = load_config_module("FFN_Rank_Reallocation", "ffn_budget_reference")
    opts = Namespace(name="audit", sequence_length=8192, data_root="/tmp/data",
                     save_folder="/tmp/checkpoints", work_dir="/tmp/work")
    for target in common.VARIANTS:
        reference = reference_module.make_build_config("f3")(opts, [])
        actual = common.make_build_config(target)(opts, [])
        reference.model = actual.model
        reference.trainer.callbacks["wandb"].group = common.variant_name(target)
        assert actual.as_dict() == reference.as_dict()
        assert actual.model.num_params == 797_050_880
        ff = actual.model.block.feed_forward.build(2048, init_device="meta")
        att = actual.model.block.sequence_mixer.build(2048, layer_idx=0, n_layers=16, init_device="meta")
        assert sum(p.numel() for p in ff.parameters()) == 10_485_760
        assert sum(p.numel() for p in att.parameters()) == 13_635_584
        for world_size in (8, 16, 32):
            assert actual.data_loader.global_batch_size % (actual.train_module.rank_microbatch_size * world_size) == 0
        actual.model.block.sequence_mixer.q_rank = 512
        with pytest.raises(ValueError, match="attention ranks"):
            common.validate_model(actual.model, target)
        actual.model.block.sequence_mixer.q_rank = 832
        actual.model.block.sequence_mixer.q_nonlinearity = common.MoSENonlinearity.identity
        with pytest.raises(ValueError, match="Q/K SiLU"):
            common.validate_model(actual.model, target)
