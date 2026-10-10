"""Check actual budgets and isolate Q/K SiLU from the registered FFN recipes."""

import importlib.util
from argparse import Namespace
from pathlib import Path

import pytest


def test_qk_reallocation_budget_and_unchanged_recipe():
    path = Path(__file__).resolve().parents[1] / "cfps/BS256_Seq8K_30B_WSD_FP8/FFN_Rank_Reallocation/_qk_common.py"
    spec = importlib.util.spec_from_file_location("ffn_qk_interaction_test", path)
    common = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(common)
    for target, primary_target in common.VARIANTS.items():
        opts = Namespace(name="audit", sequence_length=8192, data_root="/tmp/data",
                         save_folder="/tmp/checkpoints", work_dir="/tmp/work")
        reference = common.primary.make_build_config(primary_target)(opts, [])
        actual = common.make_build_config(target)(opts, [])
        reference.model.block.sequence_mixer.q_nonlinearity = common.MoSENonlinearity.silu
        reference.model.block.sequence_mixer.k_nonlinearity = common.MoSENonlinearity.silu
        reference.trainer.callbacks["wandb"].group = common.variant_name(target)
        assert actual.as_dict() == reference.as_dict()
        assert actual.model.num_params == 797_050_880
        ff = actual.model.block.feed_forward.build(2048, init_device="meta")
        att = actual.model.block.sequence_mixer.build(2048, layer_idx=0, n_layers=16, init_device="meta")
        assert sum(p.numel() for p in ff.parameters()) == 15_728_640
        assert sum(p.numel() for p in att.parameters()) == actual.model.block.sequence_mixer.num_params(2048)
        for world_size in (8, 16, 32):
            assert actual.data_loader.global_batch_size % (actual.train_module.rank_microbatch_size * world_size) == 0
        att_config = actual.model.block.sequence_mixer
        att_config.q_nonlinearity = common.MoSENonlinearity.identity
        with pytest.raises(ValueError, match="Q and K SiLU"):
            common.validate_model(actual.model, target)
