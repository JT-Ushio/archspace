import copy
import csv
import json
import tempfile
import unittest
import importlib.util
from unittest.mock import patch
from argparse import Namespace
from dataclasses import fields
from pathlib import Path

import torch
from olmo_core.nn.attention.backend import AttentionBackendName
from olmo_core.nn.transformer import TransformerConfig
from olmo_mose.svd_initialization import (SVDInitTransformerConfig, canonical_svd, make_factors,
    rms, tensor_hash, file_hash, config_fingerprint, projection_pairs, apply_artifacts)
from cfps.BS256_Seq8K_30B_WSD_FP8.Initialization_Ablations._common import build_init_config


def configuration(mode="independent"):
    opts = Namespace(sequence_length=None, data_root="/tmp/data", work_dir="/tmp/work",
                     save_folder="/tmp/checkpoints", name="test-init")
    return build_init_config(opts, [], mode)


def tiny():
    cfg = configuration().model
    cfg.d_model, cfg.n_layers, cfg.vocab_size = 16, 1, 128
    att, ff = cfg.block.sequence_mixer, cfg.block.feed_forward
    att.rank, att.n_heads, att.n_kv_heads, att.head_dim = 8, 2, 2, 8
    att.backend, att.sliding_window = AttentionBackendName.torch, None
    ff.hidden_size, ff.r2, ff.down_r2 = 64, 8, 8
    return TransformerConfig(**{f.name: getattr(cfg, f.name) for f in fields(TransformerConfig)})


def artifacts(model, cfg, root):
    pairs = {}
    for index, (name, (enc, dec)) in enumerate(projection_pairs(model).items()):
        modes, entry = make_factors(enc.weight.detach(), dec.weight.detach(), std=.02, seed=index)
        path = root / f"{name}.pt"
        torch.save(modes, path)
        entry.update(file=path.name, sha256=file_hash(path))
        pairs[name] = entry
    (root / "manifest.json").write_text(json.dumps(dict(status="complete", model_sha256=config_fingerprint(cfg), pairs=pairs)))
    (root / "probe_complete.json").write_text(json.dumps(dict(status="passed", manifest_sha256=file_hash(root / "manifest.json"))))


class InitializationTests(unittest.TestCase):
    def test_exact_svd_and_sign(self):
        torch.manual_seed(4)
        w = torch.randn(24, 16)
        e, d, s = canonical_svd(w, 8)
        u, _, vh = torch.linalg.svd(w, full_matrices=False)
        torch.testing.assert_close(d @ e, (u[:, :8] * s[:8]) @ vh[:8], atol=2e-6, rtol=2e-6)
        self.assertTrue((e[torch.arange(8), e.abs().argmax(1)] > 0).all())
        torch.testing.assert_close(e @ e.T, d.T @ d, atol=1e-5, rtol=1e-5)

    def test_scale_matching_deterministic_rng(self):
        e, d = torch.randn(8, 16)*.02, torch.randn(24, 8)*.35
        state = torch.get_rng_state().clone()
        modes, metrics = make_factors(e, d, std=.02, seed=51)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        again, _ = make_factors(e, d, std=.02, seed=51)
        for key, reference in (("E", e), ("D", d)):
            torch.testing.assert_close(rms(modes["svd_scale_matched"][key]), rms(reference))
            torch.testing.assert_close(modes["svd_raw"][key], again["svd_raw"][key], atol=0, rtol=0)
        raw, matched = modes["svd_raw"], modes["svd_scale_matched"]
        torch.testing.assert_close(matched["D"] @ matched["E"], metrics["product_scale"]*(raw["D"] @ raw["E"]))

    def test_real_model_pairing_loader_and_backward(self):
        torch.set_num_threads(2)
        cfg = tiny()
        baseline = cfg.build(init_device="cpu")
        baseline.init_weights(device=torch.device("cpu"), max_seq_len=8)
        reference = copy.deepcopy(baseline.state_dict())
        self.assertEqual(len(projection_pairs(baseline)), 7)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifacts(baseline, cfg, root)
            for mode in ("independent", "svd_scale_matched"):
                custom = SVDInitTransformerConfig(**{f.name: getattr(cfg, f.name) for f in fields(TransformerConfig)},
                                                 factor_init=mode, factor_init_artifacts=temporary)
                model = custom.build(init_device="cpu")
                model.init_weights(device=torch.device("cpu"), max_seq_len=8)
                self.assertEqual(list(reference), list(model.state_dict()))
                targets = {id(m.weight) for pair in projection_pairs(model).values() for m in pair}
                for name, value in model.named_parameters():
                    if mode == "independent" or id(value) not in targets:
                        torch.testing.assert_close(value, reference[name], atol=0, rtol=0)
                y = model(torch.randint(0, 128, (2, 8)))
                y.square().mean().backward()
                self.assertTrue(torch.isfinite(y).all())
                self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
            with self.assertRaises(ValueError):
                apply_artifacts(baseline, root, "independent", "wrong-fingerprint")
            (root / "probe_complete.json").write_text('{"status":"failed"}')
            with self.assertRaises(ValueError):
                apply_artifacts(baseline, root, "independent", config_fingerprint(cfg))

    def test_full_config_defaults_and_only_init_changes(self):
        a, c = configuration(), configuration("svd_scale_matched")
        self.assertEqual(config_fingerprint(a.model), config_fingerprint(c.model))
        self.assertEqual(a.data_loader.global_batch_size, 256*8192)
        self.assertEqual(a.train_module.rank_microbatch_size, 4*8192)
        self.assertEqual(a.train_module.optim.lr, .005)
        self.assertTrue(a.train_module.float8_config.enabled)
        self.assertEqual(a.train_module.scheduler.warmup, 2000)
        self.assertEqual(a.train_module.scheduler.decay_fraction, .2)
        self.assertEqual(a.trainer.max_duration.value, 30_000_000_000)
        self.assertEqual(a.trainer.callbacks["checkpointer"].save_interval, 1788)
        att, ff = a.model.block.sequence_mixer, a.model.block.feed_forward
        self.assertEqual(att.rank, 512)
        self.assertEqual((ff.r1, ff.r2, ff.down_r1, ff.down_r2), (0,512,0,512))
        self.assertFalse(ff.share_gate_up_subspace)
        self.assertEqual(str(att.share_scope), "none")
        for name in ("q", "k", "v", "o"):
            self.assertEqual(str(getattr(att, f"{name}_nonlinearity")), "silu")
        self.assertEqual(str(ff.gate_nonlinearity), "silu")
        self.assertEqual(str(ff.up_nonlinearity), "silu")
        self.assertEqual(str(ff.down_nonlinearity), "rms_norm")

    def test_probe_hooks_and_sampled_ce(self):
        script = Path(__file__).resolve().parents[1] / "scripts/analysis/Initialization_Analysis/probe.py"
        spec = importlib.util.spec_from_file_location("svd_probe", script)
        probe = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(probe)
        cfg = tiny()
        model = cfg.build(init_device="cpu")
        model.init_weights(device=torch.device("cpu"), max_seq_len=8)
        model.eval()
        ids, positions = torch.arange(10,18), torch.tensor([1,3,5])
        sample = dict(ids=ids, positions=positions, labels=ids[positions+1], meta=dict(sample=0, source="toy"))
        with patch.object(torch.Tensor, "cuda", lambda x,*a,**kw: x):
            acts, ce = probe.forward_probe(model, [sample], "independent", "bf16")
        self.assertEqual(len(acts), 21)
        expected = torch.nn.functional.cross_entropy(model(ids[None])[0,positions].float(), sample["labels"])
        self.assertAlmostEqual(ce[0]["ce"], expected.item(), places=5)
        self.assertTrue(all(len(m._forward_hooks)==0 for pair in projection_pairs(model).values() for m in pair))

    def test_report_rejects_missing_condition(self):
        script = Path(__file__).resolve().parents[1] / "scripts/analysis/Initialization_Analysis/probe_report.py"
        spec = importlib.util.spec_from_file_location("svd_report", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "manifest.json").write_text(json.dumps(dict(world_size=1, pairs={"blocks.0.down": {}})))
            (root / "samples.json").write_text(json.dumps(dict(samples=[dict(sample=0)])))
            (root / "rank-0.complete.json").write_text(json.dumps(dict(samples=[0], fsdp_initialization_smoke="passed")))
            ces, acts, weights = [], [], []
            for mode in ("independent", "svd_raw", "svd_scale_matched"):
                weights.append(dict(mode=mode, pair="blocks.0.down", E_rms=1, D_rms=1, DE_frobenius=1))
                for prec in ("bf16", "fp8"):
                    ces.append(dict(mode=mode, precision=prec, sample=0, ce=3, tokens=128))
                    for stage in ("a", "z", "y"):
                        acts.append(dict(mode=mode, precision=prec, sample=0, stage=stage, pair="blocks.0.down",
                                         rms=1, abs_max=2, mean=0, negative_fraction=.5, abs_p99=1, abs_p999=2))
            def save(stem, rows):
                with (root / f"{stem}-rank0.csv").open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
            save("ce", ces)
            save("activations", acts)
            save("weights", weights)
            module.report(root)
            self.assertTrue((root / "REPORT.md").exists())
            self.assertTrue((root / "fp8-down-z-rms.svg").exists())
            save("ce", ces[:-1])
            with self.assertRaises(ValueError):
                module.report(root)


if __name__ == "__main__":
    unittest.main()
