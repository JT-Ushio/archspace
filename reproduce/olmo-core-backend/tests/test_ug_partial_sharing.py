import importlib.util
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
from olmo_core.data import TokenizerConfig
from olmo_core.nn.layer_norm import RMSNorm
from olmo_core.nn.transformer.init import InitMethod

from olmo_mose import (
    MoSENonlinearity,
    MoSESwiGLUConfig,
    SwiGLUChannelControl,
    SwiGLUChannelControlScope,
)


class PartialSharingTests(unittest.TestCase):
    def test_separate_branch_sharing_matches_formula_and_gradients(self):
        torch.manual_seed(17)
        for share_linear in (False, True):
            for share_nonlinear in (False, True):
                for nonlinearity in (MoSENonlinearity.identity, MoSENonlinearity.rms_norm):
                    with self.subTest(linear=share_linear, nonlinear=share_nonlinear, fn=nonlinearity):
                        config = MoSESwiGLUConfig(
                            hidden_size=12, r1=4, r2=3, down_r1=0, down_r2=4,
                            control=SwiGLUChannelControl.standard,
                            control_scope=SwiGLUChannelControlScope.none,
                            gate_nonlinearity=nonlinearity,
                            up_nonlinearity=nonlinearity,
                            down_nonlinearity=MoSENonlinearity.rms_norm,
                            rms_norm_learnable_weight=True,
                            share_gate_up_subspace=True,
                            share_linear_gate_up_subspace=share_linear,
                            share_nonlinear_gate_up_subspace=share_nonlinear,
                        )
                        module = config.build(8)
                        InitMethod.normal.init_feed_forward(module, d_model=8, block_idx=0, num_blocks=1)
                        x = torch.randn(2, 3, 8, requires_grad=True)
                        g1 = module.linear_u(x) if share_linear else module.gate_linear_u(x)
                        u1 = g1 if share_linear else module.up_linear_u(x)
                        g2 = module.nonlinear_u(x) if share_nonlinear else module.gate_nonlinear_u(x)
                        u2 = g2 if share_nonlinear else module.up_nonlinear_u(x)
                        if nonlinearity == MoSENonlinearity.rms_norm:
                            gn = module.gate_up_nonlinear_norm if share_nonlinear else module.gate_nonlinear_norm
                            un = module.gate_up_nonlinear_norm if share_nonlinear else module.up_nonlinear_norm
                            g2, u2 = gn(g2), un(u2)
                        gate = module.gate_linear_v(g1) + module.gate_nonlinear_v(g2)
                        up = module.up_linear_v(u1) + module.up_nonlinear_v(u2)
                        expected = module.down_nonlinear_v(
                            module.down_nonlinear_norm(module.down_nonlinear_u(F.silu(gate) * up))
                        )
                        actual = module(x)
                        torch.testing.assert_close(actual, expected)
                        inputs = (x, *module.parameters())
                        expected_grads = torch.autograd.grad(expected.square().sum(), inputs, retain_graph=True)
                        actual_grads = torch.autograd.grad(actual.square().sum(), inputs)
                        for eg, ag in zip(expected_grads, actual_grads):
                            # RMSNorm uses float32 reductions; the reference
                            # evaluates shared normalization twice, changing
                            # the accumulation order of its gradient.
                            torch.testing.assert_close(eg, ag, atol=1e-4, rtol=1e-4)
                            self.assertTrue(torch.isfinite(ag).all())
                        self.assertEqual(sum(p.numel() for p in module.parameters()), config.num_params(8))
                        self.assertIsInstance(module.down_nonlinear_norm, RMSNorm)

    def test_legacy_sharing_defaults_preserve_modules(self):
        for share in (False, True):
            config = MoSESwiGLUConfig(hidden_size=12, r1=4, r2=3, share_gate_up_subspace=share)
            module = config.build(8)
            self.assertEqual(module.share_linear_gate_up_subspace, share)
            self.assertEqual(module.share_nonlinear_gate_up_subspace, share)
            self.assertEqual(sum(p.numel() for p in module.parameters()), config.num_params(8))

    def test_actual_sweep_counts_and_fixed_other_projections(self):
        path = Path(__file__).resolve().parents[1] / "cfps/BS256_Seq8K_30B_WSD_FP8/UG_Partial_Sharing/_common.py"
        spec = importlib.util.spec_from_file_location("ug_partial_test_config", path)
        common = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(common)
        baseline = None
        for target in common.VARIANTS:
            model = common.build_model(TokenizerConfig.dolma2(), target)
            expected = 797_050_880 - (2_621_440 if target == "p1" else 0)
            self.assertEqual(model.num_params, expected)
            ff = model.block.feed_forward.build(model.d_model, init_device="meta")
            self.assertEqual(sum(p.numel() for p in ff.parameters()), model.block.feed_forward.num_params(model.d_model))
            att = model.block.sequence_mixer.build(model.d_model, layer_idx=0, n_layers=16, init_device="meta")
            self.assertEqual(sum(p.numel() for p in att.parameters()), model.block.sequence_mixer.num_params(model.d_model))
            if baseline is None:
                baseline = model.block.sequence_mixer.as_dict()
            self.assertEqual(model.block.sequence_mixer.as_dict(), baseline)
            common.validate_model(model, target)
            model.block.feed_forward.down_r2 = 520
            with self.assertRaisesRegex(ValueError, "Down rank"):
                common.validate_model(model, target)
            print(target, expected)


if __name__ == "__main__":
    unittest.main()
