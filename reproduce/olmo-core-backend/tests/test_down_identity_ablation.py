"""CPU checks; also runnable directly with Python (no pytest dependency)."""
import copy
import runpy
import shlex
import sys
import unittest
from argparse import Namespace
from pathlib import Path

import torch
import torch.nn.functional as F

from olmo_mose import (
    LowRankAttentionConfig,
    MoSESwiGLU,
    MoSESwiGLUConfig,
    SwiGLUChannelControl,
)


class DownIdentityTests(unittest.TestCase):
    def make_module(self, activation):
        torch.manual_seed(42)
        return MoSESwiGLU(
            d_model=8, hidden_size=32, r1=0, r2=8,
            down_r1=0, down_r2=8, dtype=torch.float64,
            control=SwiGLUChannelControl.standard,
            share_gate_up_subspace=False, down_nonlinearity=activation,
            rms_norm_learnable_weight=False,
        )

    def test_same_parameters_and_initialization(self):
        baseline = self.make_module("silu")
        for activation in ("identity", "rms_norm", "silu_rms_norm"):
            other = self.make_module(activation)
            self.assertEqual(list(baseline.state_dict()), list(other.state_dict()))
            for key, value in baseline.state_dict().items():
                torch.testing.assert_close(value, other.state_dict()[key], rtol=0, atol=0)
            self.assertIsNone(other.down_linear_u)
            self.assertIsNone(other.down_linear_v)
            self.assertEqual(len(other.projection_modules()), 6)

    def test_identity_forward_and_backward(self):
        model = self.make_module("identity")
        reference = copy.deepcopy(model)
        x = torch.randn(2, 3, 8, dtype=torch.float64, requires_grad=True)
        xr = x.detach().clone().requires_grad_(True)
        gate = reference.gate_nonlinear_v(F.silu(reference.gate_nonlinear_u(xr)))
        up = reference.up_nonlinear_v(F.silu(reference.up_nonlinear_u(xr)))
        hidden = F.silu(gate) * up
        expected = reference.down_nonlinear_v(reference.down_nonlinear_u(hidden))
        actual = model(x)
        torch.testing.assert_close(actual, expected)
        actual.square().sum().backward()
        expected.square().sum().backward()
        torch.testing.assert_close(x.grad, xr.grad)
        for (_, p), (_, pr) in zip(model.named_parameters(), reference.named_parameters()):
            torch.testing.assert_close(p.grad, pr.grad)

    def test_silu_then_rms_forward_and_backward(self):
        model = self.make_module("silu_rms_norm")
        reference = copy.deepcopy(model)
        x = torch.randn(2, 3, 8, dtype=torch.float64, requires_grad=True)
        xr = x.detach().clone().requires_grad_(True)
        gate = reference.gate_nonlinear_v(F.silu(reference.gate_nonlinear_u(xr)))
        up = reference.up_nonlinear_v(F.silu(reference.up_nonlinear_u(xr)))
        a = reference.down_nonlinear_u(F.silu(gate) * up)
        expected = reference.down_nonlinear_v(reference.down_nonlinear_norm(F.silu(a)))
        actual = model(x)
        torch.testing.assert_close(actual, expected)
        wrong_order = reference.down_nonlinear_v(F.silu(reference.down_nonlinear_norm(a)))
        self.assertFalse(torch.allclose(actual, wrong_order))
        actual.square().sum().backward()
        expected.square().sum().backward()
        torch.testing.assert_close(x.grad, xr.grad)
        for (_, p), (_, pr) in zip(model.named_parameters(), reference.named_parameters()):
            torch.testing.assert_close(p.grad, pr.grad)

    def test_composite_norm_creation_and_parameter_counts(self):
        # The shared activation enum is also used by Gate/Up and attention.
        # Check all creation/counting paths, even though this run changes Down only.
        for gain in (False, True):
            for shared in (False, True):
                ff_config = MoSESwiGLUConfig(
                    hidden_size=32, r1=0, r2=8, down_r1=0, down_r2=8,
                    share_gate_up_subspace=shared,
                    gate_nonlinearity="silu_rms_norm", up_nonlinearity="silu_rms_norm",
                    down_nonlinearity="silu_rms_norm", rms_norm_learnable_weight=gain,
                )
                ff = ff_config.build(8)
                self.assertEqual(ff_config.num_params(8), sum(p.numel() for p in ff.parameters()))
                self.assertIsNotNone(ff.down_nonlinear_norm)
                self.assertTrue(torch.isfinite(ff(torch.randn(2, 3, 8))).all())
                attn_config = LowRankAttentionConfig(
                    n_heads=2, rank=8, bias=False,
                    share_scope="qkv" if shared else "none",
                    q_nonlinearity="silu_rms_norm", k_nonlinearity="silu_rms_norm",
                    v_nonlinearity="silu_rms_norm", o_nonlinearity="silu_rms_norm",
                    rms_norm_learnable_weight=gain,
                )
                attention = attn_config.build(8, layer_idx=0, n_layers=1)
                self.assertEqual(attn_config.num_params(8), sum(p.numel() for p in attention.parameters()))
                x = torch.randn(2, 3, 8)
                q = attention.w_q
                torch.testing.assert_close(q(x), q.u(q.nonlinear_norm(F.silu(q.v(x)))))
                self.assertEqual(
                    attention._can_share_activated_latent((attention.w_q, attention.w_k)),
                    not gain,
                )

    def test_four_configs_only_change_down_activation(self):
        folder = Path(__file__).resolve().parents[1] / "cfps/BS256_Seq8K_30B_WSD_FP8/Down_Identity_Ablations"
        old_path = sys.path.copy()
        old_common = sys.modules.pop("_common", None)
        sys.path.insert(0, str(folder))
        normalized = []
        try:
            for label, activation in (
                ("silu", "silu"), ("identity", "identity"), ("rms", "rms_norm"),
                ("silu_rms_norm", "silu_rms_norm"),
            ):
                opts = Namespace(sequence_length=None, data_root="/tmp/data", work_dir="/tmp/work",
                                 save_folder="/tmp/checkpoints", name="test-down-activation")
                entry = runpy.run_path(str(folder / f"down_{label}_fp8.py"))
                config = entry["build_config"](opts, [])
                # Parse actual launcher overrides too, without sourcing preparation
                # scripts, reading credentials, mounting data or starting training.
                launcher = folder.parents[2] / "scripts/train/Down_Identity_Ablations" / f"run_down_{label}_fp8.sh"
                flags = []
                for token in shlex.split(launcher.read_text().replace("\\\n", " ")):
                    flag = token.removeprefix("--")
                    if flag.startswith(("model.", "train_module.", "trainer.", "data_loader.")) and "=" in flag:
                        if flag.startswith("trainer.work_dir="):
                            continue
                        flags.append(flag)
                from_launcher = entry["build_config"](opts, flags)
                self.assertEqual(config.as_config_dict(), from_launcher.as_config_dict())
                ff = config.model.block.feed_forward
                self.assertEqual(str(ff.down_nonlinearity), activation)
                self.assertEqual((ff.r1, ff.r2, ff.down_r1, ff.down_r2), (0, 512, 0, 512))
                self.assertFalse(ff.share_gate_up_subspace)
                self.assertFalse(ff.rms_norm_learnable_weight)
                self.assertEqual(config.model.block.sequence_mixer.rank, 512)
                self.assertEqual(str(config.model.block.sequence_mixer.share_scope), "none")
                self.assertEqual(config.data_loader.global_batch_size, 256 * 8192)
                self.assertEqual(config.train_module.rank_microbatch_size, 32768)
                self.assertEqual(config.train_module.optim.lr, .005)
                self.assertTrue(config.train_module.float8_config.enabled)
                self.assertEqual(config.train_module.scheduler.warmup, 2000)
                self.assertEqual(config.train_module.scheduler.decay_fraction, .2)
                self.assertEqual(config.trainer.max_duration.value, 30_000_000_000)
                self.assertEqual(str(config.trainer.max_duration.unit), "tokens")
                ckpt = config.trainer.callbacks["checkpointer"]
                self.assertEqual(ckpt.save_interval, 1788)
                self.assertIsNone(ckpt.max_checkpoints)
                ff.down_nonlinearity = "silu"
                config.trainer.callbacks["wandb"].group = "comparison"
                normalized.append(config.as_config_dict())
            self.assertEqual(normalized[0], normalized[1])
            self.assertEqual(normalized[0], normalized[2])
            self.assertEqual(normalized[0], normalized[3])
        finally:
            sys.path[:] = old_path
            sys.modules.pop("_common", None)
            if old_common is not None:
                sys.modules["_common"] = old_common


if __name__ == "__main__":
    unittest.main()
