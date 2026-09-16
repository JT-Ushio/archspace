"""CPU-only invariants and actual experiment config checks; no training submission."""
import copy
import runpy
import sys
import unittest
from argparse import Namespace
from pathlib import Path

import torch
from torch.nn import functional as F
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.transformer.init import InitMethod
from olmo_mose import MoSESwiGLUConfig
from olmo_mose.output_subspace import GroupedOutputRotation
from cfps.BS256_Seq8K_30B_WSD_FP8.Down_Output_Subspace_Ablations._common import build_rotation_config


def ff_config(mode="none", **kwargs):
    settings = dict(hidden_size=32, r1=0, r2=8, down_r1=0, down_r2=8,
                    control="standard", control_scope="none", share_gate_up_subspace=False,
                    down_nonlinearity="identity", down_output_rotation=mode,
                    down_rotation_groups=2)
    return MoSESwiGLUConfig(**(settings | kwargs))


class OutputSubspaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_old_state_dict_rng_and_zero_start(self):
        torch.manual_seed(123)
        base = ff_config().build(8)
        state = torch.get_rng_state()
        x = torch.randn(2, 3, 8)
        for mode in ("static", "dynamic", "dynamic_residual"):
            torch.manual_seed(123)
            model = ff_config(mode).build(8)
            self.assertTrue(torch.equal(state, torch.get_rng_state()))
            for key, value in base.state_dict().items():
                torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)
            expected = {"output_subspace.weight"}
            if mode == "dynamic_residual":
                expected.add("output_subspace.residual_gate")
            self.assertEqual(set(model.state_dict()) - set(base.state_dict()), expected)
            self.assertEqual(len(model.projection_modules()), len(base.projection_modules()))
            torch.testing.assert_close(model(x), base(x), rtol=0, atol=0)
            self.assertEqual(ff_config(mode).num_params(8), sum(p.numel() for p in model.parameters()))
        # Default mode requires no new keys when loading a legacy checkpoint.
        ff_config().build(8).load_state_dict(base.state_dict(), strict=True)

    def test_meta_materialization_runtime_init_preserves_rng(self):
        refs = None
        for mode in ("none", "static", "dynamic", "dynamic_residual"):
            model = ff_config(mode).build(8, init_device="meta")
            model.to_empty(device="cpu")
            gen = torch.Generator().manual_seed(17)
            InitMethod.normal.init_feed_forward(model, d_model=8, block_idx=0,
                                                num_blocks=1, generator=gen)
            matrices = [m.weight.detach().clone() for m in model.projection_modules()]
            if refs is None:
                refs, rng = matrices, gen.get_state()
            else:
                self.assertTrue(torch.equal(rng, gen.get_state()))
                for a, b in zip(refs, matrices):
                    torch.testing.assert_close(a, b, atol=0, rtol=0)
                if mode == "dynamic_residual":
                    self.assertGreater(model.output_subspace.weight.count_nonzero(), 0)
                    self.assertEqual(model.output_subspace.residual_gate.count_nonzero(), 0)
                else:
                    self.assertEqual(model.output_subspace.weight.count_nonzero(), 0)

    def test_orthogonality_reference_backward_and_autocast(self):
        for mode in ("static", "dynamic"):
            rot = GroupedOutputRotation(4, 16, groups=2, mode=mode)
            torch.nn.init.normal_(rot.weight, std=.2)
            y = torch.randn(2, 3, 16, requires_grad=True)
            z = torch.randn(2, 3, 4, requires_grad=True)
            c, s = rot.coefficients(z)
            torch.testing.assert_close(c.square()+s.square(), torch.ones_like(c), atol=2e-7, rtol=2e-7)
            result = rot(y, z)
            torch.testing.assert_close(result.norm(dim=-1), y.norm(dim=-1), atol=1e-6, rtol=1e-6)
            # Independent 2x2 matrix reference for pair ordering and orientation.
            matrix = torch.stack((c, -s, s, c), -1).reshape(*c.shape, 2, 2)
            expected = torch.matmul(matrix.unsqueeze(-3), y.reshape(2, 3, 2, 4, 2).unsqueeze(-1))
            torch.testing.assert_close(result, expected.reshape_as(y))
            result.mul(torch.randn_like(result)).sum().backward()
            self.assertTrue(torch.isfinite(rot.weight.grad).all())
            self.assertGreater(rot.weight.grad.abs().sum(), 0)
            self.assertTrue(torch.isfinite(y.grad).all())
            if mode == "dynamic":
                self.assertTrue(torch.isfinite(z.grad).all())
                self.assertGreater(z.grad.abs().sum(), 0)
            with torch.autocast("cpu", dtype=torch.bfloat16):
                b = rot(y.detach().bfloat16(), z.detach().bfloat16())
                c32, _ = rot.coefficients(z.detach().bfloat16())
            self.assertEqual(c32.dtype, torch.float32)
            self.assertEqual(b.dtype, torch.bfloat16)
            torch.testing.assert_close(b.float().norm(dim=-1), y.detach().bfloat16().float().norm(dim=-1),
                                       atol=.015, rtol=.005)

    def test_gradient_at_zero_is_not_dead_and_roundtrip(self):
        for mode in ("static", "dynamic"):
            model = ff_config(mode).build(8)
            x = torch.randn(2, 3, 8)
            model(x).mul(torch.randn(2, 3, 8)).sum().backward()
            self.assertGreater(model.output_subspace.weight.grad.abs().sum(), 0)
            # Nonzero-state save/load through normal state_dict API.
            torch.nn.init.normal_(model.output_subspace.weight, std=.3)
            other = ff_config(mode).build(8)
            other.load_state_dict(model.state_dict(), strict=True)
            torch.testing.assert_close(model(x), other(x))

    def test_zero_residual_gate_stages_controller_learning(self):
        torch.manual_seed(11)
        base = ff_config().build(8)
        torch.manual_seed(11)
        model = ff_config("dynamic_residual").build(8)
        x = torch.randn(2, 3, 8)
        torch.testing.assert_close(model(x), base(x), rtol=0, atol=0)
        target = torch.randn(2, 3, 8)
        model(x).mul(target).sum().backward()
        self.assertGreater(model.output_subspace.residual_gate.grad.abs().sum(), 0)
        self.assertEqual(model.output_subspace.weight.grad.count_nonzero(), 0)

        model.zero_grad(set_to_none=True)
        with torch.no_grad():
            model.output_subspace.residual_gate.fill_(0.1)
        model(x).mul(target).sum().backward()
        self.assertGreater(model.output_subspace.weight.grad.abs().sum(), 0)

    def test_dynamic_can_leave_fixed_span_static_cannot(self):
        z = torch.tensor([[1., 0.], [1., 2.]])
        y = torch.tensor([[1., 0.], [1., 0.]])
        for mode, rank in (("static", 1), ("dynamic", 2)):
            rot = GroupedOutputRotation(2, 2, groups=1, mode=mode)
            with torch.no_grad():
                rot.weight.copy_(torch.tensor([[0., .5]]))
            self.assertEqual(torch.linalg.matrix_rank(rot(y, z)).item(), rank)

    def test_invalid_combinations(self):
        for kwargs in (dict(down_r1=1), dict(down_r2=0), dict(down_nonlinearity="silu"),
                       dict(bias=True), dict(down_rotation_groups=0), dict(down_rotation_groups=True),
                       dict(down_rotation_a_max=float("nan")), dict(down_rotation_a_max=0)):
            with self.subTest(kwargs=kwargs), self.assertRaises((OLMoConfigurationError, ValueError)):
                ff_config("dynamic", **kwargs).build(8)
        with self.assertRaises(OLMoConfigurationError):
            ff_config("dynamic", down_rotation_groups=3).build(8)

    def test_compile_forward_backward(self):
        for mode in ("static", "dynamic", "dynamic_residual"):
            model = ff_config(mode).build(8)
            torch.nn.init.normal_(model.output_subspace.weight, std=.1)
            if model.output_subspace.residual_gate is not None:
                torch.nn.init.constant_(model.output_subspace.residual_gate, .2)
            ref = copy.deepcopy(model)
            compiled = torch.compile(model, backend="aot_eager", fullgraph=True)
            x = torch.randn(2, 3, 8, requires_grad=True)
            xr = x.detach().clone().requires_grad_(True)
            y, yr = compiled(x), ref(xr)
            torch.testing.assert_close(y, yr)
            y.square().sum().backward()
            yr.square().sum().backward()
            torch.testing.assert_close(x.grad, xr.grad)
            for p, pr in zip(model.parameters(), ref.parameters()):
                torch.testing.assert_close(p.grad, pr.grad)

    def test_real_transformer_init_optimizer_and_fp8_conversion(self):
        from olmo_core.nn.attention.backend import AttentionBackendName
        from torchao.float8.float8_linear import Float8Linear
        opts = Namespace(sequence_length=None, data_root="/tmp/data", work_dir="/tmp/work",
                         save_folder="/tmp/checkpoints", name="test-output-subspace")
        original = None
        for mode in ("none", "static", "dynamic", "dynamic_residual"):
            config = build_rotation_config(opts, [], mode)
            cfg = config.model
            cfg.d_model, cfg.n_layers, cfg.vocab_size = 32, 2, 128
            att, ff = cfg.block.sequence_mixer, cfg.block.feed_forward
            att.rank, att.n_heads, att.n_kv_heads, att.head_dim = 16, 2, 2, 16
            att.backend, att.sliding_window = AttentionBackendName.torch, None
            ff.hidden_size, ff.r2, ff.down_r2, ff.down_rotation_groups = 64, 16, 16, 2
            model = cfg.build(init_device="meta")
            model.init_weights(device=torch.device("cpu"), max_seq_len=8)
            if original is None:
                original = copy.deepcopy(model.state_dict())
            for key, value in original.items():
                torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
            # A complete tiny Transformer forward/backward, not just the isolated transform.
            out = model(torch.randint(0, 128, (2, 8)))
            out.mul(torch.randn_like(out)).sum().backward()
            categories = config.train_module.optim.categorize_parameters(model)
            names = [n for n, _ in model.named_parameters() if "output_subspace" in n]
            expected_names = 0 if mode == "none" else (4 if mode == "dynamic_residual" else 2)
            self.assertEqual(len(names), expected_names)
            categorized = set().union(*map(set, categories.values()))
            for name in names:
                self.assertIn(name, categorized)
                self.assertTrue(torch.isfinite(model.get_parameter(name).grad).all())
                if mode == "dynamic_residual" and name.endswith(".weight"):
                    self.assertEqual(model.get_parameter(name).grad.count_nonzero(), 0)
                else:
                    self.assertGreater(model.get_parameter(name).grad.abs().sum(), 0)
            controllers = {n: model.get_parameter(n) for n in names}
            # Conversion itself works on CPU; this does NOT execute FP8 GPU GEMMs.
            config.train_module.float8_config.apply_float8_linear(model)
            self.assertTrue(any(isinstance(m, Float8Linear) for m in model.modules()))
            for n, p in controllers.items():
                self.assertIs(model.get_parameter(n), p)
                self.assertEqual(p.dtype, torch.float32)
                self.assertIsInstance(model.get_submodule(n.rsplit(".", 1)[0]), GroupedOutputRotation)

    def test_configs_only_rotation_diff_and_entrypoints(self):
        opts = Namespace(sequence_length=None, data_root="/tmp/data", work_dir="/tmp/work",
                         save_folder="/tmp/checkpoints", name="test-output-subspace")
        baseline = build_rotation_config(opts, [], "none").as_config_dict()
        folder = Path(__file__).resolve().parents[1] / "cfps/BS256_Seq8K_30B_WSD_FP8/Down_Output_Subspace_Ablations"
        old_common = sys.modules.pop("_common", None)
        old_path = sys.path.copy()
        sys.path.insert(0, str(folder))
        try:
            variants = (
                ("identity", "none"),
                ("static", "static"),
                ("dynamic", "dynamic"),
                ("dynamic_residual", "dynamic_residual"),
            )
            for label, mode in variants:
                c = runpy.run_path(str(folder / f"down_{label}_fp8.py"))["build_config"](opts, [])
                self.assertEqual(c.data_loader.global_batch_size, 256*8192)
                self.assertEqual(c.train_module.rank_microbatch_size, 32768)
                self.assertEqual(c.train_module.optim.lr, .005)
                self.assertTrue(c.train_module.float8_config.enabled)
                self.assertEqual(c.train_module.scheduler.warmup, 2000)
                self.assertEqual(c.train_module.scheduler.decay_fraction, .2)
                self.assertEqual(c.trainer.max_duration.value, 30_000_000_000)
                self.assertEqual(c.trainer.callbacks["checkpointer"].save_interval, 1788)
                self.assertIsNone(c.trainer.callbacks["checkpointer"].max_checkpoints)
                self.assertEqual(str(c.model.block.feed_forward.down_output_rotation), mode)
                c.model.block.feed_forward.down_output_rotation = "none"
                self.assertEqual(c.as_config_dict(), baseline)
            with self.assertRaises(OLMoConfigurationError):
                build_rotation_config(opts, ["model.block.feed_forward.down_nonlinearity=silu"], "dynamic")
        finally:
            sys.path[:] = old_path
            sys.modules.pop("_common", None)
            if old_common is not None:
                sys.modules["_common"] = old_common


if __name__ == "__main__":
    unittest.main()
