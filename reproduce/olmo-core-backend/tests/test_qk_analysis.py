"""Validate masking, probability accounting, paired scoring and config gates."""
import copy
import importlib.util
import runpy
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import torch

BACKEND = Path(__file__).resolve().parents[1]
ROOT = BACKEND / "scripts/analysis/QK_Combination_Analysis"
spec = importlib.util.spec_from_file_location("qk_analysis", ROOT / "analyze.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)
from metrics import attention_statistics
from report import build_report, write_rows


class QKAnalysisTests(unittest.TestCase):
    def test_real_model_hook_and_sampled_logits_cpu(self):
        from olmo_core.nn.attention.backend import AttentionBackendName
        opts = Namespace(sequence_length=None, data_root="/tmp/data", work_dir="/tmp/work",
                         save_folder="/tmp/checkpoints", name="test-hooks")
        entry = BACKEND / "cfps/BS256_Seq8K_30B_WSD_FP8/All_Linear_Ablation/all_linear_fp8.py"
        config = runpy.run_path(str(entry))["build_config"](opts, []).model
        config.d_model, config.n_layers, config.vocab_size = 16, 1, 128
        att, ff = config.block.sequence_mixer, config.block.feed_forward
        att.rank, att.n_heads, att.n_kv_heads, att.head_dim = 8, 2, 2, 8
        att.backend, att.sliding_window = AttentionBackendName.torch, None
        ff.hidden_size, ff.r2, ff.down_r2 = 64, 8, 8
        model = config.build(init_device="cpu")
        model.init_weights(max_seq_len=8, device=torch.device("cpu"))
        model.eval().requires_grad_(False)
        ids, positions = torch.tensor([10, 11, 12, 13, 14, 15, 16, 17]), torch.tensor([1, 3, 6])
        sample = {"ids": ids, "positions": positions, "queries": positions,
                  "labels": ids[positions + 1],
                  "meta": {"sample": 0, "valid_length": 8, "eligible_labels": 7, "source": "toy"}}
        with tempfile.TemporaryDirectory() as temporary, patch.object(torch.Tensor, "cuda", lambda x, *a, **kw: x):
            rows, tokens = analysis.run_model(model, [sample], "L", 3576, "bf16", Path(temporary))
        self.assertEqual(len(rows), 2)
        self.assertEqual(len(tokens), 3)
        with torch.no_grad():
            logits = model(ids[None, :])
            expected = torch.nn.functional.cross_entropy(logits[0, positions].float(), sample["labels"], reduction="none")
        torch.testing.assert_close(torch.tensor([r["ce"] for r in tokens]), expected)
        self.assertEqual(len(model.blocks["0"].attention.backend._forward_pre_hooks), 0)

    def test_uniform_attention_mask_and_bins(self):
        q = k = torch.zeros(1, 8, 2, 4)
        stats, bins = attention_statistics(q, k, torch.tensor([0, 3]), 5)
        self.assertEqual(stats["eligible_keys"], [2.5, 2.5])
        torch.testing.assert_close(bins.sum(-1), torch.ones(2, 2))
        self.assertAlmostEqual(stats["self_mass"][0], .625)
        self.assertAlmostEqual(stats["mean_distance"][0], .75)
        self.assertAlmostEqual(stats["normalized_entropy"][0], .5)
        # Right-padding keys 5..7 must never contribute.
        self.assertEqual(float(bins[..., 5:].sum()), 0)
        mass = sum(stats[key][0] for key in stats if key.endswith("_mass"))
        self.assertAlmostEqual(mass, 1)

    def test_sliding_window_and_scale(self):
        q = k = torch.ones(1, 8, 1, 4)
        stats, bins = attention_statistics(q, k, torch.tensor([4]), 8, (2, 0), scale=.25)
        self.assertAlmostEqual(stats["logit_mean"][0], 1)
        self.assertAlmostEqual(stats["eligible_keys"][0], 3)
        self.assertAlmostEqual(stats["mean_distance"][0], 1, places=6)
        self.assertAlmostEqual(stats["self_mass"][0], 1 / 3, places=6)
        self.assertEqual(float(bins[..., :2].sum()), 0)
        self.assertEqual(float(bins[..., 5:].sum()), 0)
        with self.assertRaises(ValueError):
            attention_statistics(q, k, torch.tensor([8]), 8)

    def test_real_configs_and_reject_mismatch(self):
        opts = Namespace(sequence_length=None, data_root="/tmp/data", work_dir="/tmp/work",
                         save_folder="/tmp/checkpoints", name="test-qk-analysis")
        root = BACKEND / "cfps/BS256_Seq8K_30B_WSD_FP8"
        paths = {"L": "All_Linear_Ablation/all_linear_fp8.py",
                 "Q": "All_Linear_Attention_SiLU_Ablations/q_only_silu_fp8.py",
                 "K": "All_Linear_Attention_SiLU_Ablations/k_only_silu_fp8.py"}
        configs = {(label, 14306): runpy.run_path(str(root / path))["build_config"](opts, []).as_config_dict()
                   for label, path in paths.items()}
        analysis.audit_configs(configs)
        bad = copy.deepcopy(configs)
        bad["Q", 14306]["model"]["block"]["feed_forward"]["down_nonlinearity"] = "silu"
        with self.assertRaises(ValueError):
            analysis.audit_configs(bad)
        bad = copy.deepcopy(configs)
        bad["K", 14306]["init_seed"] = 999
        with self.assertRaises(ValueError):
            analysis.audit_configs(bad)

    def test_sample_labels_respect_masks_and_alignment(self):
        class Dataset:
            offsets = [(0, 1)]
            paths = ["heldout.npy"]
            def prepare(self):
                pass
            def __getitem__(self, index):
                return {"input_ids": torch.tensor([10, 11, 12, 13, 0, 0]),
                        "label_mask": torch.tensor([True, False, True, True, False, False])}
        class Config:
            tokenizer = Namespace(pad_token_id=0)
            def build(self):
                return Dataset()
        raw = {"trainer": {"callbacks": {"lm_evaluator": {"eval_dataset": {"sequence_length": 8192}}}}}
        args = Namespace(work_dir=Path("/tmp/qk-test"), data_root=None, num_sequences=1,
                         seed=42, tokens_per_sequence=128, queries=32)
        with patch.object(analysis.NumpyPaddedFSLDatasetConfig, "from_dict", return_value=Config()):
            samples, metadata, _ = analysis.prepare_samples(raw, args)
        self.assertEqual(samples[0]["positions"].tolist(), [1, 2])
        self.assertEqual(samples[0]["labels"].tolist(), [12, 13])
        self.assertEqual(metadata[0]["valid_length"], 4)

    def test_aggregate_paired_ce_and_refuse_duplicate_assignments(self):
        import json
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = {"world_size": 1, "arguments": {"steps": [14306]}}
            samples = [{"sample": i, "positions": [1, 2], "source": "heldout"} for i in range(2)]
            (root / "manifest.json").write_text(json.dumps(manifest))
            (root / "samples.json").write_text(json.dumps(samples))
            (root / "rank-00.complete.json").write_text(json.dumps({"samples": [0, 1]}))
            worker = root / "rank-00"
            worker.mkdir()
            for label, ce in (("L", 2.), ("Q", 1.8), ("K", 1.9)):
                for precision in ("bf16", "fp8"):
                    tokens = [{"sample": s, "position": p, "target_id": p+10,
                               "ce": ce + (.01 if precision == "fp8" else 0)}
                              for s in range(2) for p in (1, 2)]
                    attention = [{"sample": s, "layer": 0, "head": 0,
                                  **{metric: .5 for metric in ("normalized_entropy", "entropy", "mean_distance", "logit_std", "max_probability", "distance_over1024_mass")}}
                                 for s in range(2)]
                    write_rows(worker / f"{label}-14306-{precision}-tokens.csv", tokens)
                    write_rows(worker / f"{label}-14306-{precision}-attention.csv", attention)
            summaries, effects = build_report(root)
            self.assertEqual(len(summaries), 6)
            self.assertAlmostEqual(next(r["mean_delta_ce"] for r in effects if r["comparison"] == "L_minus_Q"), .2)
            self.assertAlmostEqual(next(r["mean_delta_ce"] for r in effects if r["comparison"] == "Q_fp8_minus_bf16"), .01)
            self.assertTrue((root / "plots/ce-bf16.svg").is_file())
            (root / "rank-00.complete.json").write_text(json.dumps({"samples": [0, 0]}))
            with self.assertRaises(ValueError):
                build_report(root)


if __name__ == "__main__":
    unittest.main()
