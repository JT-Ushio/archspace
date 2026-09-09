"""CPU tests for offline analysis; no real training checkpoints or credentials."""
import copy
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from olmo_core.nn.layer_norm import RMSNorm
from olmo_mose import MoSESwiGLU, SwiGLUChannelControl

BACKEND = Path(__file__).resolve().parents[1]
FOLDER = BACKEND / "scripts/analysis/Down_Checkpoint_Analysis"
sys.path.insert(0, str(FOLDER))
import analyze_checkpoints as analysis
from plot_results import build_plots


class ToyBlock(nn.Module):
    def __init__(self, activation):
        super().__init__()
        self.feed_forward = MoSESwiGLU(
            d_model=8, hidden_size=32, r1=0, r2=8, down_r1=0, down_r2=8,
            share_gate_up_subspace=False, down_nonlinearity=activation,
            control=SwiGLUChannelControl.standard,
        )
        self.feed_forward_norm = RMSNorm(size=8, bias=False)

    def forward(self, x):
        return x + self.feed_forward_norm(self.feed_forward(x))


class ToyModel(nn.Module):
    def __init__(self, activation):
        super().__init__()
        self.blocks = nn.ModuleDict({"0": ToyBlock(activation), "1": ToyBlock(activation)})

    def forward(self, x):
        for block in self.blocks.values():
            x = block(x)
        return x


class CheckpointAnalysisTests(unittest.TestCase):
    def test_validation_sampling_excludes_padding(self):
        import numpy as np
        from cached_path import set_cache_dir
        from olmo_core.data import NumpyPaddedFSLDatasetConfig, TokenizerConfig
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            set_cache_dir(root / "cache")
            tokens = np.array([2, 3, 4, 5, 1] * 6, dtype=np.uint16)
            tokens.tofile(root / "validation.bin")
            cfg = NumpyPaddedFSLDatasetConfig(
                tokenizer=TokenizerConfig(vocab_size=32, eos_token_id=1, pad_token_id=0),
                paths=[str(root / "validation.bin")], sequence_length=8192, work_dir=str(root),
            )
            raw = {"trainer": {"callbacks": {"lm_evaluator": {"eval_dataset": cfg.as_config_dict()}}}}
            args = Namespace(work_dir=root, data_root=None, num_sequences=4, seed=34521, tokens_per_sequence=128)
            # Keep this tiny unit test from spawning a host-sized process pool.
            with patch("olmo_core.data.numpy_dataset.concurrent.futures.ProcessPoolExecutor",
                       new=partial(ThreadPoolExecutor, max_workers=2)):
                samples, manifest, _ = analysis.get_samples(raw, args)
            self.assertEqual(len(samples), 4)
            for (ids, positions), record in zip(samples, manifest):
                self.assertEqual(tuple(ids.shape), (1, 8192))
                self.assertTrue(ids[0, positions].ne(0).all())
                self.assertEqual(len(positions), record["valid_tokens"])
                self.assertLess(record["valid_tokens"], 8192)

    def test_selection_and_latest_checkpoint_fail_closed(self):
        indices = analysis.pick_indices([(0, 10), (10, 100)], 12, 42)
        self.assertEqual(indices, analysis.pick_indices([(0, 10), (10, 100)], 12, 42))
        self.assertEqual(len(set(indices)), 12)
        self.assertEqual(sum(i < 10 for i in indices), 6)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "step10/model_and_optim").mkdir(parents=True)
            (root / "step10/model_and_optim/.metadata").touch()
            (root / "step10/config.json").write_text("{}")
            self.assertEqual(analysis.resolve_checkpoint(root)[1], 10)
            (root / "step20").mkdir()
            with self.assertRaises(FileNotFoundError):
                analysis.resolve_checkpoint(root)

    def test_stats_and_replay(self):
        x = torch.randn(128, 8)
        stats = analysis.ActivationStats(covariance=True)
        stats.update(x[:64])
        stats.update(x[64:])
        result, _, spectrum = stats.finish()
        cov = torch.cov(x.T, correction=0)
        expected = torch.linalg.eigvalsh(cov).flip(0).clamp_min(0)
        expected /= expected.sum()
        torch.testing.assert_close(spectrum.float(), expected, atol=1e-6, rtol=1e-5)
        self.assertEqual(result["sampled_tokens"], 128)
        self.assertAlmostEqual(result["rms_mean"], x.square().mean(-1).sqrt().mean().item(), places=6)
        block = ToyBlock("rms_norm")
        a = torch.randn(128, 8)
        out = analysis.replay(a, block.feed_forward.down_nonlinear_v, block.feed_forward_norm)
        torch.testing.assert_close(out["R"]["z"], analysis.rms_reference(a, 1e-5))
        torch.testing.assert_close(out["SR"]["z"], analysis.rms_reference(torch.nn.functional.silu(a), 1e-5))
        torch.testing.assert_close(out["I"]["o"], out["R"]["o"], atol=2e-4, rtol=2e-4)
        zero = analysis.distances(torch.zeros(2, 8), torch.zeros(2, 8))
        self.assertEqual(len(zero["one_minus_cosine"]), 0)
        self.assertTrue(zero["undefined_cosine"].eq(1).all())

    def test_hooks_do_not_modify_forward_and_generate_outputs(self):
        torch.manual_seed(123)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            for label, activation in analysis.ACTIVATIONS.items():
                model = ToyModel(activation)
                x = torch.randn(1, 16, 8)
                expected = model(x)
                collector = analysis.Collector(model)
                collector.positions = torch.tensor([0, 3, 7, 9])
                with torch.inference_mode():
                    actual = model(x)
                torch.testing.assert_close(actual, expected)
                collector.close()
                self.assertEqual(set(collector.stats), {(l, p) for l in (1, 2) for p in analysis.POINTS})
                self.assertTrue(all(v.count == 4 for v in collector.stats.values()))
                (directory / label).mkdir()
                collector.save(directory / label, label, 1)
            analysis.write_json(directory / "manifest.json", {"valid_tokens": 16, "sampled_tokens_per_layer": 4})
            build_plots(directory)
            self.assertTrue((directory / "report.md").is_file())
            self.assertGreater(len(list((directory / "plots").glob("*.svg"))), 30)

    def test_saved_config_audit_and_dcp_model_only_reload(self):
        folder = BACKEND / "scripts/analysis/Down_Identity_Ablations/results_four_runs"
        configs = {label: json.loads((folder / f"{label}_metadata.json").read_text())["config"] for label in analysis.ACTIVATIONS}
        analysis.audit_configs(configs)
        wrong = copy.deepcopy(configs)
        wrong["R"]["train_module"]["optim"]["lr"] = 1e-3
        with self.assertRaises(ValueError):
            analysis.audit_configs(wrong)
        # Use a real serialized model class and real DCP reader/writer, but tiny
        # tensors in /tmp. Deliberately include optimizer state in the checkpoint.
        from olmo_core.nn.transformer import TransformerConfig
        from olmo_core.distributed.checkpoint import save_model_and_optim_state
        cfg = TransformerConfig.from_dict(copy.deepcopy(configs["S"]["model"]))
        cfg.d_model, cfg.n_layers, cfg.vocab_size = 8, 2, 32
        cfg.block.feed_forward.hidden_size = 32
        cfg.block.feed_forward.r2 = cfg.block.feed_forward.down_r2 = 8
        cfg.block.sequence_mixer.rank = 8
        cfg.block.sequence_mixer.n_heads = 2
        model = cfg.build(init_device="cpu")
        optim = torch.optim.AdamW(model.parameters())
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        optim.step()
        with tempfile.TemporaryDirectory() as tmp:
            from cached_path import set_cache_dir
            set_cache_dir(Path(tmp) / "cache")
            checkpoint = Path(tmp) / "step1"
            checkpoint.mkdir()
            save_model_and_optim_state(checkpoint / "model_and_optim", model, optim)
            restored = analysis.load_model({"model": cfg.as_config_dict()}, checkpoint, torch.device("cpu"))
            for name, value in model.state_dict().items():
                torch.testing.assert_close(restored.state_dict()[name], value.bfloat16(), atol=0, rtol=0)
            with torch.inference_mode():
                logits = restored(torch.randint(0, 32, (1, 8)), logits_to_keep=1)
            self.assertEqual(tuple(logits.shape), (1, 1, 32))


if __name__ == "__main__":
    unittest.main()
