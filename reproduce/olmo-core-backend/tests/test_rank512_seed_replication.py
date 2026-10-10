"""Replications must match the actual original recipes, not only their budget."""

import importlib.util
import sys
from argparse import Namespace
from pathlib import Path

import pytest


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_seed_replication_preserves_original_config():
    root = Path(__file__).resolve().parents[1] / "cfps/BS256_Seq8K_30B_WSD_FP8"
    common = load(root / "Rank512_Seed_Replication/_common.py", "seed_replication_test")
    filenames = {"c0": "c0_k_silu_down_rms.py", "c2": "c2_k_silu_down_rms_gateup.py",
                 "c6": "c6_qk_silu_down_rms_gateup.py"}
    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(sys.modules, "_common", common.reference)
        originals = {base: load(root / "Rank512Matched_Combinations" / name, base + "_original_test")
                     for base, name in filenames.items()}
    for target, (base, seed) in common.VARIANTS.items():
        opts = Namespace(name="audit", sequence_length=8192, data_root="/tmp/data",
                         save_folder="/tmp/checkpoints", work_dir="/tmp/work")
        original = originals[base].build_config(opts, [])
        actual = common.make_build_config(target)(opts, [])
        original.init_seed = seed
        original.trainer.callbacks["wandb"].group = common.variant_name(target)
        assert actual.as_dict() == original.as_dict()
        assert actual.model.num_params == 797_050_880
        att = actual.model.block.sequence_mixer.build(2048, layer_idx=0, n_layers=16, init_device="meta")
        ff = actual.model.block.feed_forward.build(2048, init_device="meta")
        # Q/K head normalization adds4096 weights beyond the projection budget.
        assert sum(p.numel() for p in att.parameters()) + sum(p.numel() for p in ff.parameters()) == 24_117_248 + 4096
        for world_size in (8, 16, 32):
            assert actual.data_loader.global_batch_size % (actual.train_module.rank_microbatch_size * world_size) == 0
        with pytest.raises(ValueError, match="initialization/data seeds"):
            common.make_build_config(target)(opts, ["init_seed=12536"])
        with pytest.raises(ValueError, match="original C0/C2/C6 architecture"):
            common.make_build_config(target)(opts, ["model.block.sequence_mixer.k_nonlinearity=identity"])
