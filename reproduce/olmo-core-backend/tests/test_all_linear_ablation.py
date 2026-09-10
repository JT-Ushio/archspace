"""Check all low-rank Identity projections without removing the SwiGLU gate."""
import copy
import runpy
import shlex
import unittest
from argparse import Namespace
from pathlib import Path

import torch
import torch.nn.functional as F

from olmo_mose import LowRankAttentionConfig, MoSESwiGLU, SwiGLUChannelControl

BACKEND = Path(__file__).resolve().parents[1]


class AllLinearTests(unittest.TestCase):
    def test_ffn_forward_backward_and_initialization(self):
        def build(activation):
            torch.manual_seed(42)
            return MoSESwiGLU(
                d_model=8, hidden_size=32, r1=0, r2=8,
                down_r1=0, down_r2=8, dtype=torch.float64,
                control=SwiGLUChannelControl.standard,
                share_gate_up_subspace=False,
                gate_nonlinearity=activation, up_nonlinearity=activation,
                down_nonlinearity=activation, rms_norm_learnable_weight=False,
            )
        model = build("identity")
        baseline = build("silu")
        self.assertEqual(list(model.state_dict()), list(baseline.state_dict()))
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, baseline.state_dict()[name], rtol=0, atol=0)
        reference = copy.deepcopy(model)
        x = torch.randn(2, 3, 8, dtype=torch.float64, requires_grad=True)
        xr = x.detach().clone().requires_grad_(True)
        gate = reference.gate_nonlinear_v(reference.gate_nonlinear_u(xr))
        up = reference.up_nonlinear_v(reference.up_nonlinear_u(xr))
        expected = reference.down_nonlinear_v(reference.down_nonlinear_u(F.silu(gate) * up))
        actual = model(x)
        torch.testing.assert_close(actual, expected)
        without_gate_silu = reference.down_nonlinear_v(reference.down_nonlinear_u(gate * up))
        self.assertFalse(torch.allclose(actual, without_gate_silu))
        actual.square().sum().backward()
        expected.square().sum().backward()
        torch.testing.assert_close(x.grad, xr.grad)
        for p, pr in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(p.grad, pr.grad)

    def test_qkvo_identity_forward_backward(self):
        config = LowRankAttentionConfig(
            n_heads=2, rank=8, bias=False, share_scope="none",
            q_nonlinearity="identity", k_nonlinearity="identity",
            v_nonlinearity="identity", o_nonlinearity="identity",
        )
        model = config.build(8, layer_idx=0, n_layers=1)
        self.assertEqual(config.num_params(8), sum(p.numel() for p in model.parameters()))
        for name in ("w_q", "w_k", "w_v", "w_out"):
            projection = getattr(model, name)
            self.assertIsNone(projection.nonlinear_norm)
            reference = copy.deepcopy(projection)
            x = torch.randn(2, 3, 8, requires_grad=True)
            xr = x.detach().clone().requires_grad_(True)
            actual, expected = projection(x), reference.u(reference.v(xr))
            torch.testing.assert_close(actual, expected)
            actual.square().sum().backward()
            expected.square().sum().backward()
            torch.testing.assert_close(x.grad, xr.grad)
            for p, pr in zip(projection.parameters(), reference.parameters()):
                torch.testing.assert_close(p.grad, pr.grad)

    def test_config_launcher_and_baseline_match(self):
        entry = runpy.run_path(str(BACKEND / "cfps/BS256_Seq8K_30B_WSD_FP8/All_Linear_Ablation/all_linear_fp8.py"))
        opts = Namespace(sequence_length=None, data_root="/tmp/data", work_dir="/tmp/work",
                         save_folder="/tmp/checkpoints", name="test-all-linear")
        config = entry["build_config"](opts, [])
        folder = BACKEND / "scripts/train/All_Linear_Ablation"
        self.assertEqual({p.name for p in folder.glob("*.sh")},
                         {"run_all_linear_fp8.sh", "run_ffn_linear_attention_silu_fp8.sh"})
        flags = []
        for token in shlex.split((folder / "run_all_linear_fp8.sh").read_text().replace("\\\n", " ")):
            flag = token.removeprefix("--")
            if flag.startswith(("model.", "train_module.", "trainer.", "data_loader.")) and "=" in flag:
                if not flag.startswith("trainer.work_dir="):
                    flags.append(flag)
        launched = entry["build_config"](opts, flags)
        self.assertEqual(config.as_config_dict(), launched.as_config_dict())
        from cfps.BS256_Seq8K_30B_WSD_FP8.Down_Identity_Ablations._common import build_down_activation_config
        from olmo_mose import MoSENonlinearity
        baseline = build_down_activation_config(
            opts, [], variant="all-lowrank-linear-fp8-30b",
            down_nonlinearity=MoSENonlinearity.silu,
        )
        for name in ("q", "k", "v", "o"):
            self.assertEqual(str(getattr(config.model.block.sequence_mixer, f"{name}_nonlinearity")), "identity")
            setattr(config.model.block.sequence_mixer, f"{name}_nonlinearity", MoSENonlinearity.silu)
        for name in ("gate", "up", "down"):
            self.assertEqual(str(getattr(config.model.block.feed_forward, f"{name}_nonlinearity")), "identity")
            setattr(config.model.block.feed_forward, f"{name}_nonlinearity", MoSENonlinearity.silu)
        self.assertEqual(config.as_config_dict(), baseline.as_config_dict())

    def test_ffn_linear_attention_silu_config(self):
        folder = BACKEND / "cfps/BS256_Seq8K_30B_WSD_FP8/All_Linear_Ablation"
        entry = runpy.run_path(str(folder / "ffn_linear_attention_silu_fp8.py"))
        opts = Namespace(sequence_length=None, data_root="/tmp/data", work_dir="/tmp/work",
                         save_folder="/tmp/checkpoints", name="test-ffn-linear")
        config = entry["build_config"](opts, [])
        launcher = BACKEND / "scripts/train/All_Linear_Ablation/run_ffn_linear_attention_silu_fp8.sh"
        flags = []
        for token in shlex.split(launcher.read_text().replace("\\\n", " ")):
            flag = token.removeprefix("--")
            if flag.startswith(("model.", "train_module.", "trainer.", "data_loader.")) and "=" in flag:
                if not flag.startswith("trainer.work_dir="):
                    flags.append(flag)
        self.assertEqual(config.as_config_dict(), entry["build_config"](opts, flags).as_config_dict())
        baseline = runpy.run_path(str(folder / "all_linear_fp8.py"))["build_config"](opts, [])
        from olmo_mose import MoSENonlinearity
        for name in ("q", "k", "v", "o"):
            self.assertEqual(str(getattr(config.model.block.sequence_mixer, f"{name}_nonlinearity")), "silu")
            setattr(baseline.model.block.sequence_mixer, f"{name}_nonlinearity", MoSENonlinearity.silu)
        baseline.trainer.callbacks["wandb"].group = config.trainer.callbacks["wandb"].group
        self.assertEqual(config.as_config_dict(), baseline.as_config_dict())
        for name in ("gate", "up", "down"):
            self.assertEqual(str(getattr(config.model.block.feed_forward, f"{name}_nonlinearity")), "identity")
        self.assertEqual(config.model.block.feed_forward.r2, 512)
        self.assertEqual(config.model.block.feed_forward.down_r2, 512)
        self.assertEqual(config.model.block.sequence_mixer.rank, 512)
        self.assertEqual(config.data_loader.global_batch_size, 2097152)
        self.assertEqual(config.train_module.rank_microbatch_size, 32768)
        for world_size in (16, 32):
            self.assertEqual(config.data_loader.global_batch_size % (world_size * 32768), 0)


if __name__ == "__main__":
    unittest.main()
